"""M1 离线单测：不依赖网络/不需要 API Key，锁住图的结构与关键逻辑。

真实 LLM 调用是非确定性的，放到 M8 的评测 harness 里用录制回放处理；
这里只测"可离线、可复现"的纯逻辑部分。
"""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage

from app.agent.graph import build_graph, should_continue
from app.agent.tools import ALL_TOOLS
from app.agent.tools.flights import search_flights


class _FakeLLM:
    """假 LLM：build_graph 只需要它能 bind_tools / with_structured_output，
    编译图时不会真正调用（M2+ 起 triage 节点会要后者）。"""

    def bind_tools(self, tools):  # noqa: ANN001
        return self

    def with_structured_output(self, schema):  # noqa: ANN001
        return self


def test_search_flights_is_deterministic():
    args = {"origin": "北京", "destination": "上海", "date": "2026-06-24"}
    a = search_flights.invoke(args)
    b = search_flights.invoke(args)
    assert a == b  # 同输入同输出 → 可复现
    data = json.loads(a)
    assert len(data["flights"]) == 3
    prices = [f["price"] for f in data["flights"]]
    assert prices == sorted(prices)  # 按价格升序


def test_should_continue_routing():
    # 有 tool_calls → 去 tools
    ai_with_call = AIMessage(
        content="",
        tool_calls=[{"name": "search_flights", "args": {}, "id": "call_1"}],
    )
    assert should_continue({"messages": [ai_with_call]}) == "tools"
    # 纯文本回答 → 结束
    assert should_continue({"messages": [AIMessage(content="好的")]}) == "end"


def test_graph_compiles_with_expected_nodes():
    graph = build_graph(_FakeLLM(), ALL_TOOLS, checkpointer=None)
    nodes = set(graph.get_graph().nodes)
    assert {"agent", "tools"} <= nodes


def test_human_message_constructs():
    # 守护：消息类型 import 路径稳定
    assert HumanMessage(content="hi").content == "hi"
