"""M6b 测试：长期记忆。

1. 离线：抽取候选的**闸门与路由**（置信度过滤、preference→表 / fact→向量库）——用替身，不碰 DB/网络；
2. 集成：偏好 upsert（recency wins）+ 语义记忆去重/召回 + 跨租户隔离（需 Postgres + Qdrant）。
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agent import memory
from app.agent.memory import ExtractedMemories, MemoryCandidate
from app.core.config import settings

# ---------- 离线：抽取的闸门与路由 ----------


def test_last_turn_and_text():
    msgs = [
        HumanMessage(content="第一轮", id="1"),
        AIMessage(content="回应1", id="2"),
        HumanMessage(content="第二轮", id="3"),
        AIMessage(content="回应2", id="4"),
    ]
    recent = memory._last_turn(msgs)
    assert [m.content for m in recent] == ["第二轮", "回应2"]  # 只取最近一轮
    assert "用户: 第二轮" in memory._turn_text(recent)


class _FakeExtractor:
    def __init__(self, mems):
        self._mems = mems

    async def ainvoke(self, messages):  # noqa: ANN001
        return ExtractedMemories(memories=self._mems)


class _FakeLLM:
    def __init__(self, mems):
        self._mems = mems

    def with_structured_output(self, schema):  # noqa: ANN001
        return _FakeExtractor(self._mems)


async def test_memorize_gates_and_routes(monkeypatch):
    """置信度低于闸门被丢；带 key 的 preference 进偏好表，其余进向量库。"""
    prefs, facts = [], []

    async def fake_upsert(tenant_id, user_id, key, value, confidence, source):  # noqa: ANN001
        prefs.append((key, value, source))

    async def fake_remember(text, tenant_id, user_id, dedup_threshold):  # noqa: ANN001
        facts.append(text)
        return True

    monkeypatch.setattr(memory, "upsert_preference", fake_upsert)
    monkeypatch.setattr(memory, "remember_semantic", fake_remember)

    mems = [
        MemoryCandidate(kind="preference", key="seat_preference", value="靠窗", confidence=0.95),
        MemoryCandidate(kind="preference", key="", value="没有 key 的偏好", confidence=0.95),
        MemoryCandidate(kind="fact", value="常出差到成都", confidence=0.8),
        MemoryCandidate(kind="fact", value="低置信应被过滤", confidence=0.3),
    ]
    msgs = [HumanMessage(content="我以后只坐靠窗"), AIMessage(content="好的，记住了")]
    stats = await memory.memorize(_FakeLLM(mems), msgs, tenant_id="t", user_id="u")

    assert prefs == [("seat_preference", "靠窗", "explicit")]  # 仅带 key 的高置信 preference
    assert "常出差到成都" in facts
    assert "没有 key 的偏好" in facts  # preference 缺 key → 退化为自由事实
    assert all("低置信" not in f for f in facts)  # 0.3 < 0.6 闸门 → 丢弃
    assert stats == {"candidates": 4, "preferences": 1, "facts": 2}


# ---------- 集成：偏好 upsert / 语义去重召回 / 隔离 ----------


@pytest.fixture
async def ids():
    """解析真实 (acme→alice, globex→bob) 的 tenant/user id；连不上 Postgres 则跳过。"""
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import create_async_engine

    from app.db.models import Tenant, User

    memory._sessionmaker.cache_clear()  # 让记忆用的引擎在本测试的事件循环里新建
    try:
        engine = create_async_engine(settings.database_url)
        async with engine.connect() as conn:
            trows = dict((await conn.execute(select(Tenant.slug, Tenant.id))).all())
            if "acme" not in trows or "globex" not in trows:
                pytest.skip("需要 seed 的租户（make seed）")
            alice = (
                await conn.execute(select(User.id).where(User.email == "alice@acme.com"))
            ).first()
            bob = (
                await conn.execute(select(User.id).where(User.email == "bob@globex.com"))
            ).first()
        await engine.dispose()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"需要 Postgres（make up + seed）：{e!r}")
    yield {
        "acme": str(trows["acme"]),
        "globex": str(trows["globex"]),
        "alice": str(alice[0]),
        "bob": str(bob[0]),
    }
    memory._sessionmaker.cache_clear()


async def test_preference_upsert_recency_wins(ids):
    acme, alice, globex, bob = ids["acme"], ids["alice"], ids["globex"], ids["bob"]

    # 首次写 → 读回
    await memory.upsert_preference(acme, alice, "seat_preference", "靠窗", 0.9, "explicit")
    prefs = await memory.load_preferences(acme, alice)
    assert prefs.get("seat_preference") == "靠窗"

    # 同 key 改值 → 覆盖（recency wins），不产生第二条
    await memory.upsert_preference(acme, alice, "seat_preference", "靠过道", 0.95, "explicit")
    prefs = await memory.load_preferences(acme, alice)
    assert prefs.get("seat_preference") == "靠过道"
    assert list(prefs.keys()).count("seat_preference") == 1

    # 跨租户隔离：globex 上下文里读 bob 的偏好，绝不含 acme/alice 的那条（RLS）
    other = await memory.load_preferences(globex, bob)
    assert "靠过道" not in other.values()


async def test_semantic_memory_dedup_and_recall(ids):
    from qdrant_client import models as qm

    from app.infra.qdrant import ensure_memory_collection, get_qdrant_client

    get_qdrant_client.cache_clear()
    client = get_qdrant_client()
    try:
        await client.get_collections()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"需要 Qdrant（make up）：{e!r}")
    acme, alice = ids["acme"], ids["alice"]
    await ensure_memory_collection(client)
    sel = qm.FilterSelector(filter=memory._mem_filter(acme, alice))
    await client.delete(collection_name=settings.memory_collection, points_selector=sel)

    try:
        # 首次写入成功；语义相同再写 → 去重返回 False
        assert await memory.remember_semantic("用户经常出差到成都分公司", acme, alice, 0.9) is True
        assert await memory.remember_semantic("用户经常出差到成都分公司", acme, alice, 0.9) is False

        # 召回：相关 query 命中
        hits = await memory.recall_semantic("成都出差", acme, alice, k=3, threshold=0.4)
        assert any("成都" in h for h in hits)
    finally:
        await client.delete(collection_name=settings.memory_collection, points_selector=sel)
        await client.close()
        get_qdrant_client.cache_clear()
