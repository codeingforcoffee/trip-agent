"""M5 测试：RAG 切块（离线）+ 检索的租户隔离（集成，需 Qdrant + 首次会下载嵌入模型）。

离线部分永远跑（纯逻辑）；集成部分在 Qdrant 不可用时自动 skip。
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from qdrant_client import models as qm

from app.core.config import settings
from app.infra.qdrant import ensure_rag_collection, get_qdrant_client
from app.llm.embeddings import embed_documents
from app.rag.chunk import chunk_markdown
from app.rag.retriever import retrieve

# ---------- 离线：结构感知切块 ----------

_SAMPLE = """# 测试政策

## 住宿标准

### 一线城市

一线城市住宿费报销上限为每晚 800 元。

### 二线城市

二线城市住宿费报销上限为每晚 500 元。

## 交通标准

高铁二等座据实报销，一等座需审批。
"""


def test_chunk_carries_heading_path():
    """每块都带'来源 > 章节路径'前缀，且 section 字段层级正确。"""
    chunks = chunk_markdown(_SAMPLE, source="测试政策", max_chars=450, overlap=60)
    sections = {c.section for c in chunks}
    assert "住宿标准 > 一线城市" in sections
    assert "住宿标准 > 二线城市" in sections
    assert "交通标准" in sections
    # 文本前缀含 source（标题增强：孤立块也知道自己讲什么）
    assert all(c.text.startswith("测试政策") for c in chunks)
    # 一线城市那块应包含 800、不含 500（切块边界正确，没把两节混在一起）
    one_line = next(c for c in chunks if c.section == "住宿标准 > 一线城市")
    assert "800" in one_line.text and "500" not in one_line.text


def test_chunk_long_section_splits():
    """超长小节会被按 max_chars 切成多块（防单块过大稀释相关性）。"""
    long_md = "# T\n\n## S\n\n" + ("这是一段政策说明。\n\n" * 100)
    chunks = chunk_markdown(long_md, source="T", max_chars=120, overlap=20)
    assert len(chunks) > 1
    assert all(len(c.text) <= 120 + len("T > S：") + 30 for c in chunks)  # 大致受 max 约束


# ---------- 集成：检索的租户前过滤（跨租户拿不到对方语料）----------


@pytest.fixture
async def rag_collection():
    """用一个一次性集合做隔离测试，结束删除，不污染真实 `rag`。"""
    # lru_cache 单例可能绑在别的事件循环上，先清掉，确保在本测试的 loop 里新建
    get_qdrant_client.cache_clear()
    client = get_qdrant_client()
    try:
        await client.get_collections()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"需要 Qdrant（make up）：{e!r}")
    name = f"rag_test_{uuid4().hex[:8]}"
    orig = settings.rag_collection
    settings.rag_collection = name
    await ensure_rag_collection(client)
    yield client, name
    settings.rag_collection = orig
    await client.delete_collection(name)
    await client.close()
    get_qdrant_client.cache_clear()


async def _upsert(client, name: str, tenant_id: str, texts: list[str]) -> None:
    vecs = await embed_documents(texts)
    points = [
        qm.PointStruct(
            id=str(uuid4()),
            vector=v,
            payload={"tenant_id": tenant_id, "source": "测试政策", "section": "住宿", "text": t},
        )
        for t, v in zip(texts, vecs, strict=True)
    ]
    await client.upsert(collection_name=name, points=points)


async def test_retrieval_is_tenant_isolated(rag_collection):
    client, name = rag_collection
    acme, globex = str(uuid4()), str(uuid4())
    await _upsert(client, name, acme, ["一线城市住宿费报销上限为每晚 800 元。"])
    await _upsert(client, name, globex, ["一线城市住宿费报销上限为每晚 550 元。"])

    # acme 只应看到自己的 800，绝不串到 globex 的 550
    acme_hits = await retrieve("住宿费报销上限多少", tenant_id=acme)
    assert acme_hits, "应检索到本租户内容"
    acme_text = " ".join(h.text for h in acme_hits)
    assert "800" in acme_text and "550" not in acme_text

    # globex 反之
    globex_hits = await retrieve("住宿费报销上限多少", tenant_id=globex)
    globex_text = " ".join(h.text for h in globex_hits)
    assert "550" in globex_text and "800" not in globex_text

    # 不存在的租户 → 前过滤后 0 命中（fail-closed）
    none_hits = await retrieve("住宿费报销上限多少", tenant_id=str(uuid4()))
    assert none_hits == []
