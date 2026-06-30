"""RAG 读路径（M5）：把查询向量化 → 按租户前过滤 → top-k 检索。

多租户向量隔离的关键就在这里：
  - **前过滤（pre-filter）**：把 tenant_id 过滤条件交给 Qdrant，在检索时就只在本租户的向量里
    算近邻。绝不能"检索完再丢掉别租户的结果"（post-filter）——那样 top-k 名额可能被别租户占满，
    过滤完所剩无几；更是一条**跨租户数据泄漏**红线。这是 M3 纵深防御在向量层的延伸。
  - tenant_id 来自可信的 config.configurable（M3 注入的身份），**绝不**来自 LLM 传入的工具参数，
    所以模型即便被注入也无法越权查别家语料。
  - **分数阈值**：低于阈值视为"没检索到"，让上层工具据实返回"未找到"，而不是硬塞不相关内容
    诱导模型编造——这是 faithfulness（答案忠于检索内容）的第一道闸。
"""

from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import models as qm

from app.core.config import settings
from app.core.logging import get_logger
from app.infra.qdrant import get_qdrant_client
from app.llm.embeddings import embed_query

log = get_logger("app.rag.retriever")


@dataclass(frozen=True)
class RetrievedChunk:
    """一条检索命中：正文 + 来源/章节（生成引用用）+ 相似度分数。"""

    text: str
    source: str
    section: str
    score: float


async def retrieve(
    query: str,
    *,
    tenant_id: str,
    k: int | None = None,
    score_threshold: float | None = None,
) -> list[RetrievedChunk]:
    """在本租户语料里检索与 query 最相关的 k 个文档块。"""
    k = k if k is not None else settings.rag_top_k
    threshold = score_threshold if score_threshold is not None else settings.rag_score_threshold

    qv = await embed_query(query)
    client = get_qdrant_client()
    # query_points：新接口（替代已弃用的 search）。query_filter 实现租户前过滤。
    resp = await client.query_points(
        collection_name=settings.rag_collection,
        query=qv,
        query_filter=qm.Filter(
            must=[qm.FieldCondition(key="tenant_id", match=qm.MatchValue(value=tenant_id))]
        ),
        limit=k,
        score_threshold=threshold,  # 低于阈值的命中直接不返回
        with_payload=True,
    )
    hits = [
        RetrievedChunk(
            text=p.payload.get("text", ""),
            source=p.payload.get("source", ""),
            section=p.payload.get("section", ""),
            score=p.score,
        )
        for p in resp.points
    ]
    log.info(
        "rag.retrieve",
        tenant_id=tenant_id,
        k=k,
        hits=len(hits),
        top_score=round(hits[0].score, 3) if hits else None,
    )
    return hits
