"""长期记忆（M6b）：跨会话记住用户偏好/事实，并在新会话里召回、应用。

区别于其它三样"记忆"：
  - 短期记忆(M1 checkpointer) = 这通对话的原始消息；工作记忆(triage 槽位) = 本任务槽位；
  - 上下文压缩(M6a) = 把旧消息摘要进 summary；
  - **长期记忆(本模块) = 跨对话、跨 thread 的用户画像**，按 (tenant,user) 隔离。

两个存储，各司其职（面试要点）：
  - **结构化偏好 → Postgres user_preferences**：按 (tenant,user,key) 唯一。偏好会变（靠窗→靠过道），
    结构化 + upsert 让"更新"幂等（recency wins），绝不产生自相矛盾的两条。
  - **自由事实 → Qdrant memory 集合**：情景/语义记忆，写前按 embedding 相似度去重，召回按相似度 top-k。

写入时机：turn 结束由 memorize 抽取候选、过置信度闸门后落库（当前同步实现；生产可改 write-behind
异步不阻塞用户——这里从简，代码里注明）。读取时机：每轮入口 recall 把偏好 + 相关记忆注入 system。

判定"该不该记"：只记稳定、可复用、明确/强暗示的偏好或事实；不记本次行程参数(目的地/日期)、
一次性请求、寒暄。confidence 区分"明说"(高)与"推断"(低)，避免一次行为过拟合成规则。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from functools import lru_cache
from typing import Literal

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from pydantic import BaseModel, Field
from qdrant_client import models as qm
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.core.config import settings
from app.core.logging import get_logger
from app.db.models import UserPreference
from app.db.session import set_tenant_context
from app.infra.qdrant import ensure_memory_collection, get_qdrant_client
from app.llm.embeddings import embed_documents, embed_query

log = get_logger("app.agent.memory")


# ============================ 抽取（LLM）============================


class MemoryCandidate(BaseModel):
    """一条候选记忆。kind=preference 用结构化 key（可更新）；kind=fact 是自由文本（进向量库）。"""

    kind: Literal["preference", "fact"] = "fact"
    key: str = ""  # 仅 preference：归一化键，如 seat_preference
    value: str
    confidence: float = 0.0  # 0~1；明说给 0.9+，仅从行为推断给 0.5~0.7


class ExtractedMemories(BaseModel):
    memories: list[MemoryCandidate] = Field(default_factory=list)


_EXTRACT_PROMPT = """你从下面这轮对话中，抽取值得【长期记住】的用户偏好或稳定事实（可跨会话复用）。

只记：稳定、可复用、且用户明确表达或强烈暗示的偏好/事实。
不要记：本次行程的具体参数（目的地/日期/人数）、一次性请求、寒暄闲聊、随时会变的临时信息。

对每条给出：
- kind："preference"（结构化偏好，给出归一化英文 key，尽量取自：seat_preference / cabin_class /
  preferred_airline / hotel_preference / transport_preference / meal_preference / budget_sensitivity /
  home_city / departure_time_preference / employee_level）；或 "fact"（自由事实，key 留空）。
- value：简洁中文值。
- confidence：0~1。用户明说给 0.9 以上；仅从一次行为推断给 0.5~0.7。

没有值得长期记住的，就返回空列表。

【本轮对话】
{turn}"""


def _turn_text(messages: list[AnyMessage]) -> str:
    """把一轮消息里的用户/助手发言拼成文本（跳过工具 JSON，减少噪声）。"""
    lines = []
    for m in messages:
        role = (
            "用户" if isinstance(m, HumanMessage) else "助手" if isinstance(m, AIMessage) else None
        )
        if role and m.content:
            lines.append(f"{role}: {m.content}")
    return "\n".join(lines)


def _last_turn(messages: list[AnyMessage]) -> list[AnyMessage]:
    """取最近一轮：从最后一条 HumanMessage 到结尾。"""
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return messages[i:]
    return []


async def extract_memories(llm, recent: list[AnyMessage]) -> list[MemoryCandidate]:
    """用结构化输出从最近一轮抽取候选记忆。llm 由调用方注入（便于测试替身）。"""
    extractor = llm.with_structured_output(ExtractedMemories)
    result: ExtractedMemories = await extractor.ainvoke(
        [HumanMessage(content=_EXTRACT_PROMPT.format(turn=_turn_text(recent)))]
    )
    return result.memories


# ============================ 结构化偏好（Postgres）============================


@lru_cache(maxsize=1)
def _sessionmaker():
    """给记忆用的会话工厂（懒建单例）。用 app 角色（受 RLS），供图/CLI（无 app.state）使用。

    与 RAG 的 qdrant 单例同理：API 进程里会与 app.state 的池并存一个小池，无伤大雅；
    M9 可改为经 config 复用 app.state 的池。测试用 _sessionmaker.cache_clear() 重置。
    """
    from app.db.session import build_engine, build_sessionmaker

    return build_sessionmaker(build_engine())


async def load_preferences(tenant_id: str, user_id: str) -> dict[str, str]:
    """读出某用户的全部结构化偏好（RLS 按租户过滤 + 显式 user_id 过滤）。"""
    uid = uuid.UUID(str(user_id))
    async with _sessionmaker()() as s:
        await set_tenant_context(s, tenant_id)  # 事务级 RLS 上下文
        rows = (
            await s.execute(
                select(UserPreference.key, UserPreference.value).where(
                    UserPreference.user_id == uid
                )
            )
        ).all()
        await s.rollback()  # 只读，回滚结束事务、清掉租户上下文
    return {k: v for k, v in rows}


async def upsert_preference(
    tenant_id: str, user_id: str, key: str, value: str, confidence: float, source: str
) -> None:
    """写入/更新一条偏好。冲突键 (tenant,user,key) → 覆盖（recency wins）。"""
    async with _sessionmaker()() as s:
        await set_tenant_context(s, tenant_id)
        stmt = pg_insert(UserPreference).values(
            tenant_id=uuid.UUID(str(tenant_id)),
            user_id=uuid.UUID(str(user_id)),
            key=key,
            value=value,
            confidence=confidence,
            source=source,
        )
        stmt = stmt.on_conflict_do_update(
            constraint="uq_user_preferences_tenant_user_key",
            set_={
                "value": stmt.excluded.value,
                "confidence": stmt.excluded.confidence,
                "source": stmt.excluded.source,
                "updated_at": func.now(),
            },
        )
        await s.execute(stmt)
        await s.commit()


# ============================ 语义/情景记忆（Qdrant）============================


def _mem_filter(tenant_id: str, user_id: str) -> qm.Filter:
    return qm.Filter(
        must=[
            qm.FieldCondition(key="tenant_id", match=qm.MatchValue(value=str(tenant_id))),
            qm.FieldCondition(key="user_id", match=qm.MatchValue(value=str(user_id))),
        ]
    )


async def recall_semantic(
    query: str, tenant_id: str, user_id: str, k: int, threshold: float
) -> list[str]:
    """按当前 query 语义召回该用户的历史记忆（tenant+user 前过滤）。集合不存在等异常→空。"""
    client = get_qdrant_client()
    try:
        qv = await embed_query(query)
        resp = await client.query_points(
            collection_name=settings.memory_collection,
            query=qv,
            query_filter=_mem_filter(tenant_id, user_id),
            limit=k,
            score_threshold=threshold,
            with_payload=True,
        )
        return [p.payload.get("text", "") for p in resp.points]
    except Exception as e:  # noqa: BLE001 —— 集合未建/连接异常时降级为"无记忆"
        log.info("memory.recall_semantic_skip", error=repr(e))
        return []


async def remember_semantic(
    text: str, tenant_id: str, user_id: str, dedup_threshold: float
) -> bool:
    """写入一条自由事实，先按相似度去重。写入返回 True，判为重复返回 False。"""
    client = get_qdrant_client()
    await ensure_memory_collection(client)
    vec = (await embed_documents([text]))[0]
    dup = await client.query_points(
        collection_name=settings.memory_collection,
        query=vec,
        query_filter=_mem_filter(tenant_id, user_id),
        limit=1,
        with_payload=False,
    )
    if dup.points and dup.points[0].score >= dedup_threshold:
        return False  # 与已有记忆几乎相同 → 不重复写
    await client.upsert(
        collection_name=settings.memory_collection,
        points=[
            qm.PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    "tenant_id": str(tenant_id),
                    "user_id": str(user_id),
                    "text": text,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        ],
    )
    return True


# ============================ 对图暴露的两个入口 ============================


async def recall(query: str, tenant_id: str, user_id: str) -> str:
    """读路径：偏好（总是）+ 语义记忆（按 query）→ 拼成注入 system 的记忆上下文。"""
    prefs = await load_preferences(tenant_id, user_id)
    sem = await recall_semantic(
        query, tenant_id, user_id, settings.memory_recall_k, settings.memory_recall_threshold
    )
    parts = []
    if prefs:
        parts.append("已知用户偏好：" + "；".join(f"{k}={v}" for k, v in prefs.items()))
    if sem:
        parts.append("相关历史记忆：" + "；".join(sem))
    return "\n".join(parts)


async def memorize(llm, messages: list[AnyMessage], tenant_id: str, user_id: str) -> dict:
    """写路径：抽取最近一轮的候选 → 过置信度闸门 → 偏好 upsert / 事实去重写入。返回统计。"""
    recent = _last_turn(messages)
    if not recent:
        return {}
    candidates = await extract_memories(llm, recent)
    n_pref = n_fact = 0
    for c in candidates:
        if c.confidence < settings.memory_min_confidence:
            continue  # 置信度不够，宁缺毋滥
        value = c.value.strip()
        if not value:
            continue
        if c.kind == "preference" and c.key.strip():
            source = "explicit" if c.confidence >= 0.9 else "inferred"
            await upsert_preference(tenant_id, user_id, c.key.strip(), value, c.confidence, source)
            n_pref += 1
        elif await remember_semantic(value, tenant_id, user_id, settings.memory_dedup_threshold):
            n_fact += 1
    return {"candidates": len(candidates), "preferences": n_pref, "facts": n_fact}
