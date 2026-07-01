"""Qdrant 向量库接入（M5）：客户端 + RAG 集合管理。

集合 `rag` 的设计（面试要能说清每个选择的理由）：
  - **向量维度** = BGE 输出维（512）；**距离** = Cosine。BGE 按余弦相似度训练，必须配 Cosine，
    选错度量（如欧氏）召回会莫名其妙地差。
  - **payload** 存 tenant_id / source / section / text：检索时按 tenant_id 过滤，并用 text 拼上下文、
    用 source/section 生成引用。
  - **给 tenant_id 建 payload 索引**：多租户检索每次都要带租户过滤，建索引才能走索引而非全量扫——
    租户/文档一多，这是性能的关键，也是"多租户向量库"的必备工程动作。
"""

from __future__ import annotations

from functools import lru_cache

from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qm

from app.core.config import settings
from app.core.logging import get_logger
from app.llm.embeddings import embedding_dim

log = get_logger("app.infra.qdrant")


def build_qdrant_client() -> AsyncQdrantClient:
    """工厂：建一个 Qdrant 异步客户端（lifespan / 显式管理生命周期的地方用）。"""
    return AsyncQdrantClient(url=settings.qdrant_url)


@lru_cache(maxsize=1)
def get_qdrant_client() -> AsyncQdrantClient:
    """进程内共享的 Qdrant 客户端（懒建单例）。

    给【没有 app.state 的调用方】用：RAG 工具（在图里跑，拿不到 app.state）、ingest 脚本、CLI。
    懒建——首次调用时（已在事件循环内）才创建，绕开"无事件循环时建 async 客户端"的坑。

    取舍说明：API 进程的 /health 仍用 lifespan 里那个客户端，于是 API 进程里会有两个轻量
    HTTP 客户端连同一个 Qdrant——无伤大雅。M9 可改为把 app.state.qdrant 经 config 注入给工具，
    彻底复用同一个。这里优先让工具/脚本"自带连接、能独立从 CLI 跑通"。
    """
    return build_qdrant_client()


async def _ensure_collection(client: AsyncQdrantClient, name: str, index_fields: list[str]) -> None:
    """确保某集合存在（512 维 + Cosine），并给指定 payload 字段建关键字索引（幂等）。"""
    if not await client.collection_exists(name):
        await client.create_collection(
            collection_name=name,
            vectors_config=qm.VectorParams(
                size=embedding_dim(),
                distance=qm.Distance.COSINE,  # 配 BGE 归一化向量
            ),
        )
        log.info("qdrant.collection_created", collection=name, dim=embedding_dim())
    for field in index_fields:
        try:
            await client.create_payload_index(
                collection_name=name,
                field_name=field,
                field_schema=qm.PayloadSchemaType.KEYWORD,
            )
        except Exception as e:  # noqa: BLE001 —— 索引已存在等情况不应中断
            log.info("qdrant.payload_index_skip", collection=name, field=field, error=repr(e))


async def ensure_rag_collection(client: AsyncQdrantClient) -> None:
    """确保 `rag` 集合与 tenant_id 索引存在（幂等，ingest 启动时调）。"""
    await _ensure_collection(client, settings.rag_collection, ["tenant_id"])


async def ensure_memory_collection(client: AsyncQdrantClient) -> None:
    """确保 `memory` 集合存在（M6b 长期记忆）；按 tenant_id + user_id 过滤，故两者都建索引。"""
    await _ensure_collection(client, settings.memory_collection, ["tenant_id", "user_id"])
