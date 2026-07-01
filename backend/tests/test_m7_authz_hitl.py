"""M7a 测试：工具授权 + 高危动作 HITL + 审计。

分两层：
  1. 离线（无 DB/Redis/网络）——授权名单、_run_tool_call 的 fail-closed 授权门、图级别的
     HITL 批准/拒绝/无权三条链路（假 LLM + 假高危工具 + 假审计）；
  2. 集成（需 Redis）——真实 book_trip 的幂等（同一意图重放返回同一订单号）。
"""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

import app.agent.graph as graph_mod
from app.agent.graph import build_graph
from app.core.security import SCOPE_BOOKING
from app.security import authz

# ---------- 假替身 ----------


@tool("book_trip")
async def fake_book(city: str) -> str:
    """下单替身（不碰 Redis），返回固定标记便于断言执行与否。"""
    return "FAKE_ORDER_OK"


class _FakeAgent:
    """历史里出现过 ToolMessage 就收尾，否则发起一次 book_trip 调用。"""

    async def ainvoke(self, messages):  # noqa: ANN001
        if any(isinstance(m, ToolMessage) for m in messages):
            return AIMessage(content="已处理完成。")
        return AIMessage(
            content="", tool_calls=[{"name": "book_trip", "args": {"city": "北京"}, "id": "call1"}]
        )


class _FakeLLM:
    def bind_tools(self, tools):  # noqa: ANN001
        return _FakeAgent()


def _patch_audit(monkeypatch) -> list[dict]:
    """把 record_audit 换成收集器，断言"记了哪些安全事件"，不落库。"""
    audits: list[dict] = []

    async def fake_record(tenant_id, user_id, action, detail):  # noqa: ANN001
        audits.append({"action": action, "detail": detail})

    monkeypatch.setattr(graph_mod, "record_audit", fake_record)
    return audits


def _graph(*, hitl: bool):
    return build_graph(
        _FakeLLM(),
        [fake_book],
        enable_triage=False,
        enable_compress=False,
        enable_memory=False,
        enable_hitl=hitl,
        checkpointer=MemorySaver(),  # interrupt 必须有 checkpointer 持久化中断态
    )


def _cfg(thread: str, scopes: list[str]) -> dict:
    return {
        "configurable": {
            "thread_id": thread,
            "tenant_id": "t",
            "user_id": "u",
            "scopes": scopes,
        }
    }


def _tool_texts(state: dict) -> list[str]:
    return [m.content for m in state["messages"] if isinstance(m, ToolMessage)]


# ---------- 纯授权名单 ----------


def test_authz_registry():
    assert authz.required_scope("book_trip") == SCOPE_BOOKING
    assert authz.required_scope("search_flights") is None  # 只读工具无需 scope
    assert authz.is_high_risk("book_trip") is True
    assert authz.is_high_risk("search_flights") is False
    assert authz.has_required_scope("book_trip", ["booking:write"]) is True
    assert authz.has_required_scope("book_trip", ["chat:write"]) is False  # fail-closed
    assert authz.has_required_scope("search_flights", []) is True  # 无需授权恒放行


# ---------- 工具层授权门（_run_tool_call） ----------


async def test_run_tool_call_denies_without_scope(monkeypatch):
    audits = _patch_audit(monkeypatch)
    call = {"name": "book_trip", "args": {"city": "北京"}, "id": "c1"}
    msg = await graph_mod._run_tool_call(call, {"book_trip": fake_book}, _cfg("x", ["chat:write"]))
    assert "无权限" in msg.content  # fail-closed
    assert "FAKE_ORDER_OK" not in msg.content  # 根本没执行
    assert any(a["action"] == "tool.denied" for a in audits)


async def test_run_tool_call_allows_with_scope(monkeypatch):
    audits = _patch_audit(monkeypatch)
    call = {"name": "book_trip", "args": {"city": "北京"}, "id": "c1"}
    msg = await graph_mod._run_tool_call(
        call, {"book_trip": fake_book}, _cfg("x", ["booking:write"])
    )
    assert "FAKE_ORDER_OK" in msg.content
    assert any(a["action"] == "tool.executed" for a in audits)


# ---------- 图级别 HITL 三条链路 ----------


async def test_hitl_unauthorized_denied_without_confirm(monkeypatch):
    """无 booking scope：不弹确认（不必确认注定被拒的动作），直接在工具层 fail-closed。"""
    audits = _patch_audit(monkeypatch)
    g = _graph(hitl=True)
    out = await g.ainvoke(
        {"messages": [HumanMessage(content="订机票")]}, _cfg("deny", ["chat:write"])
    )
    assert "__interrupt__" not in out  # 未中断
    assert any("无权限" in t for t in _tool_texts(out))
    assert not any("FAKE_ORDER_OK" in t for t in _tool_texts(out))
    assert any(a["action"] == "tool.denied" for a in audits)


async def test_hitl_approve_executes(monkeypatch):
    """有权 + 批准：图在 confirm 处中断 → resume(approved) → 执行 → 审计 tool.executed。"""
    audits = _patch_audit(monkeypatch)
    g = _graph(hitl=True)
    cfg = _cfg("approve", ["chat:write", "booking:write"])
    first = await g.ainvoke({"messages": [HumanMessage(content="订机票")]}, cfg)
    assert "__interrupt__" in first  # 暂停等待人工确认
    resumed = await g.ainvoke(Command(resume={"approved": True}), cfg)
    assert any("FAKE_ORDER_OK" in t for t in _tool_texts(resumed))
    assert any(a["action"] == "tool.executed" for a in audits)


async def test_hitl_reject_blocks_execution(monkeypatch):
    """有权 + 拒绝：补"已取消"的 ToolMessage、绝不执行、审计 hitl.rejected。"""
    audits = _patch_audit(monkeypatch)
    g = _graph(hitl=True)
    cfg = _cfg("reject", ["chat:write", "booking:write"])
    await g.ainvoke({"messages": [HumanMessage(content="订机票")]}, cfg)
    rejected = await g.ainvoke(Command(resume={"approved": False}), cfg)
    texts = _tool_texts(rejected)
    assert any("已拒绝" in t for t in texts)
    assert not any("FAKE_ORDER_OK" in t for t in texts)  # 没执行
    assert any(a["action"] == "hitl.rejected" for a in audits)


async def test_hitl_disabled_skips_confirm(monkeypatch):
    """关掉 HITL：高危工具不弹确认直接执行（授权仍生效，这里有权）。"""
    _patch_audit(monkeypatch)
    g = _graph(hitl=False)
    out = await g.ainvoke(
        {"messages": [HumanMessage(content="订机票")]}, _cfg("nohitl", ["booking:write"])
    )
    assert "__interrupt__" not in out
    assert any("FAKE_ORDER_OK" in t for t in _tool_texts(out))


# ---------- 集成：真实下单的幂等（需 Redis）----------


async def test_book_trip_idempotent(monkeypatch):
    from app.agent.tools import booking
    from app.infra.redis_client import get_redis_client

    get_redis_client.cache_clear()  # 换事件循环 → 重建 client
    client = get_redis_client()
    try:
        await client.ping()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"需要 Redis（make up）：{e!r}")

    cfg = {"configurable": {"tenant_id": "itest", "user_id": "u1"}}
    args = {"trip_type": "flight", "item_id": "CA1831", "date": "2026-07-10", "price": 1200}
    idem = booking._idem_key("itest", "u1", "flight", "CA1831", "2026-07-10")
    rkey = booking._result_key("itest", idem)
    await client.delete(rkey)  # 保证首跑走"创建"分支
    try:
        r1 = json.loads(await booking.book_trip.ainvoke(args, cfg))
        r2 = json.loads(await booking.book_trip.ainvoke(args, cfg))  # 幂等重放
        assert r1["order_id"] == r2["order_id"]  # 同一意图 → 同一订单号，未重复下单
        assert r1["status"] == "confirmed"
    finally:
        await client.delete(rkey)
        await client.aclose()
        get_redis_client.cache_clear()
