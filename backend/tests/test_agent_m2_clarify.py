"""M2+ 离线单测：分诊(triage) + 澄清(clarify) 路由。

不联网：路由全是纯函数；图的端到端用「假结构化 LLM」喂预设意图来驱动，验证
「缺槽→反问、槽齐→放行」两条路径，以及开关 enable_triage 的建图差异。
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from app.agent.graph import build_graph
from app.agent.tools import ALL_TOOLS
from app.agent.triage import (
    build_clarify_question,
    clarification_needs,
    compute_missing_slots,
    route_after_triage,
)
from app.agent.triage import TripIntent


# ---------- 纯函数（路由逻辑核心） ----------


def test_compute_missing_slots():
    assert compute_missing_slots(
        "flight", {"origin": "北京", "destination": None, "date": None}
    ) == ["destination", "date"]
    assert (
        compute_missing_slots("flight", {"origin": "北京", "destination": "上海", "date": "明天"})
        == []
    )


def test_clarification_needs_concrete_intent_missing_slots():
    needs = clarification_needs("weather", {"origin": None, "destination": "上海", "date": None})
    assert needs == ["date"]  # 天气只缺日期


def test_clarification_needs_ambiguous_travel():
    # "我想去上海"：unknown + 提了目的地 → 先问诉求，再补缺的槽（目的地已知不再问）
    needs = clarification_needs("unknown", {"origin": None, "destination": "上海", "date": None})
    assert needs[0] == "purpose"
    assert "origin" in needs and "date" in needs
    assert "destination" not in needs


def test_clarification_needs_passes_through():
    # 不需要槽位的意图、以及毫无线索的 unknown，都直接放行
    assert clarification_needs("policy", {"origin": None, "destination": None, "date": None}) == []
    assert clarification_needs("chitchat", {}) == []
    assert clarification_needs("unknown", {"origin": None, "destination": None, "date": None}) == []


def test_build_clarify_question():
    q = build_clarify_question(["origin", "date"])
    assert "出发" in q and "哪一天" in q
    assert build_clarify_question([])  # 兜底分支非空


def test_route_after_triage():
    assert route_after_triage({"clarify_needs": ["origin"]}) == "clarify"
    assert route_after_triage({"clarify_needs": []}) == "agent"
    assert route_after_triage({}) == "agent"  # 没有该字段也不报错


# ---------- 图的端到端（假结构化 LLM 驱动） ----------


class _FakeStructured:
    def __init__(self, intent: TripIntent):
        self._intent = intent

    async def ainvoke(self, messages):  # noqa: ANN001
        return self._intent


class _FakeAgentRunnable:
    def __init__(self, response: AIMessage):
        self._response = response

    async def ainvoke(self, messages):  # noqa: ANN001
        return self._response


class _FakeLLM:
    """triage 走结构化输出返回预设意图；agent 走 bind_tools 返回预设 AIMessage。"""

    def __init__(self, intent: TripIntent, agent_response: AIMessage | None = None):
        self._intent = intent
        self._agent_response = agent_response or AIMessage(content="(agent 已被调用)")

    def bind_tools(self, tools):  # noqa: ANN001
        return _FakeAgentRunnable(self._agent_response)

    def with_structured_output(self, schema):  # noqa: ANN001
        return _FakeStructured(self._intent)


def test_graph_nodes_toggle_with_enable_triage():
    fake = _FakeLLM(TripIntent(intent="chitchat"))
    on = set(build_graph(fake, ALL_TOOLS, enable_triage=True).get_graph().nodes)
    assert {"triage", "clarify", "agent", "tools"} <= on
    off = set(build_graph(fake, ALL_TOOLS, enable_triage=False).get_graph().nodes)
    assert "triage" not in off and "clarify" not in off


async def test_graph_clarifies_when_underspecified():
    """'我想去上海' → triage 判定含糊 → clarify 反问 → 本轮无工具调用、直接结束。"""
    fake = _FakeLLM(TripIntent(intent="unknown", destination="上海"))
    graph = build_graph(fake, ALL_TOOLS, enable_triage=True)
    out = await graph.ainvoke({"messages": [HumanMessage(content="我想去上海")]})
    last = out["messages"][-1]
    assert isinstance(last, AIMessage)
    assert not last.tool_calls  # 没有瞎调工具
    assert "出发" in last.content  # 反问里问了出发地
    assert out["clarify_needs"]  # 记录了待澄清点（可观测）


async def test_graph_proceeds_when_slots_complete():
    """槽位齐 → triage 放行 → 走 agent（这里 agent 返回纯文本，结束）。"""
    fake = _FakeLLM(
        TripIntent(intent="flight", origin="北京", destination="上海", date="2026-06-25"),
        agent_response=AIMessage(content="这就为你查询。"),
    )
    graph = build_graph(fake, ALL_TOOLS, enable_triage=True)
    out = await graph.ainvoke({"messages": [HumanMessage(content="明天北京到上海机票")]})
    assert out["clarify_needs"] == []
    assert out["messages"][-1].content == "这就为你查询。"
