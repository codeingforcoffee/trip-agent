"""RAG 写路径（M5）：把样例政策文档灌进 Qdrant。

运行：cd backend && uv run python -m app.rag.ingest    （或 make ingest）

流程：解析 → 结构感知切块 → 本地 BGE 向量化 → 带 tenant_id 写入 Qdrant 集合 `rag`。

两个工程要点：
  - **按租户写**：每个租户一份文档，point 的 payload 带该租户的 tenant_id（UUID）。
    tenant_id 从 Postgres 的 tenants 表按 slug 查出来（用超级用户连接，管理操作专用）——
    与 seed.py 一致：管理用超级用户、应用运行时用 trip_app。
  - **幂等可重跑**：用确定性 point id（uuid5 派生自 tenant_id+序号）覆盖写，并在写前按租户
    删旧点清除残留。这样调完切块参数可直接重灌，不会留垃圾。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import models as qm
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine

from app.core.config import settings
from app.core.logging import get_logger, setup_logging
from app.db.models import Tenant
from app.infra.qdrant import ensure_rag_collection, get_qdrant_client
from app.llm.embeddings import embed_documents
from app.rag.chunk import chunk_markdown

log = get_logger("app.rag.ingest")

_DATA_DIR = Path(__file__).parent / "data"

# 租户 slug → (文档展示名作 source, 文件名)。内容刻意不同，用于演示租户隔离。
_DOCS: dict[str, tuple[str, str]] = {
    "acme": ("Acme 差旅报销管理办法", "acme_policy.md"),
    "globex": ("Globex 出行费用政策", "globex_policy.md"),
}


async def _tenant_ids_by_slug() -> dict[str, str]:
    """用超级用户连库（无视 RLS），把 slug → tenant_id(UUID 字符串) 映射查出来。"""
    engine = create_async_engine(settings.database_url)
    try:
        async with engine.connect() as conn:
            rows = (await conn.execute(select(Tenant.slug, Tenant.id))).all()
            return {slug: str(tid) for slug, tid in rows}
    finally:
        await engine.dispose()


async def _ingest_one(client, tenant_id: str, source: str, text: str) -> int:
    """切块 + 向量化 + 覆盖写入某租户的文档，返回写入块数。"""
    chunks = chunk_markdown(
        text,
        source=source,
        max_chars=settings.rag_chunk_max_chars,
        overlap=settings.rag_chunk_overlap,
    )
    vectors = await embed_documents([c.text for c in chunks])

    # 幂等：先删本租户旧点（按 payload 过滤），避免改了切块后残留陈旧块
    await client.delete(
        collection_name=settings.rag_collection,
        points_selector=qm.FilterSelector(
            filter=qm.Filter(
                must=[qm.FieldCondition(key="tenant_id", match=qm.MatchValue(value=tenant_id))]
            )
        ),
    )
    points = [
        qm.PointStruct(
            # 确定性 id：同租户同序号永远同 id → 重灌即覆盖
            id=str(uuid5(NAMESPACE_URL, f"{tenant_id}:{i}")),
            vector=vec,
            payload={
                "tenant_id": tenant_id,  # 检索前过滤就靠它
                "source": c.source,
                "section": c.section,
                "text": c.text,
            },
        )
        for i, (c, vec) in enumerate(zip(chunks, vectors, strict=True))
    ]
    await client.upsert(collection_name=settings.rag_collection, points=points)
    return len(points)


async def main() -> None:
    setup_logging()
    client = get_qdrant_client()
    await ensure_rag_collection(client)
    slug_to_id = await _tenant_ids_by_slug()

    if not slug_to_id:
        print("未找到任何租户。请先 `make migrate && make seed` 建租户。")
        return

    total = 0
    for slug, (source, fname) in _DOCS.items():
        tid = slug_to_id.get(slug)
        if tid is None:
            print(f"[跳过] 租户 slug={slug} 不存在（先 seed）")
            continue
        text = (_DATA_DIR / fname).read_text(encoding="utf-8")
        n = await _ingest_one(client, tid, source, text)
        total += n
        print(f"[写入] 租户 {slug:7s} ({tid}) ← 《{source}》 切成 {n} 块")
    print(f"\n完成：共写入 {total} 块到 Qdrant 集合 '{settings.rag_collection}'。")
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
