"""M2 离线单测：工具层 + 并发执行节点。

全部不依赖网络/不需要 API Key——只测「可离线、可复现」的纯逻辑：
  - 六件套工具都注册了、且确定性（同输入同输出）；
  - pydantic args_schema 把脏参数挡在执行前（校验失败抛错）；
  - _run_tool_call 的容错：未知工具 / 参数错误都回灌成 ToolMessage 而非崩溃；
  - asyncio.gather 并发执行保持顺序、tool_call_id 正确配对。
"""

from __future__ import annotations

import asyncio
import json

import pytest
from langchain_core.messages import ToolMessage

from app.agent.graph import _run_tool_call
from app.agent.tools import ALL_TOOLS, TOOLS_BY_NAME


def test_all_six_tools_registered():
    names = {t.name for t in ALL_TOOLS}
    assert names == {
        "search_flights",
        "search_hotels",
        "search_trains",
        "get_weather",
        "query_travel_policy",
        "estimate_expense",
    }
    # ALL_TOOLS 与 TOOLS_BY_NAME 必须同源、不掉项
    assert set(TOOLS_BY_NAME) == names


def test_tools_are_deterministic():
    """同输入同输出——这是 M8 离线评测可复现的地基。

    注：query_travel_policy 自 M5 起改为 RAG 检索（异步 + 需 Qdrant + 租户身份），不再属于
    "离线确定性"工具，故不在此列；它的行为由 test_m5_rag.py 的集成测试覆盖。
    """
    cases = [
        ("search_flights", {"origin": "北京", "destination": "上海", "date": "2026-06-24"}),
        ("search_hotels", {"city": "上海", "checkin_date": "2026-06-24", "nights": 2}),
        ("search_trains", {"origin": "北京", "destination": "上海", "date": "2026-06-24"}),
        ("get_weather", {"city": "上海", "date": "2026-06-24"}),
        (
            "estimate_expense",
            {"city": "上海", "days": 3, "hotel_price_per_night": 800, "transport_cost": 1200},
        ),
    ]
    for name, args in cases:
        tool = TOOLS_BY_NAME[name]
        assert tool.invoke(args) == tool.invoke(args), f"{name} 不是确定性的"


def test_results_shape():
    """抽查几个工具的返回结构与排序约定。"""
    hotels = json.loads(
        TOOLS_BY_NAME["search_hotels"].invoke(
            {"city": "上海", "checkin_date": "2026-06-24", "nights": 2}
        )
    )
    assert len(hotels["hotels"]) == 3
    prices = [h["price_per_night"] for h in hotels["hotels"]]
    assert prices == sorted(prices)  # 按每晚价格升序
    assert hotels["hotels"][0]["total_price"] == hotels["hotels"][0]["price_per_night"] * 2

    exp = json.loads(
        TOOLS_BY_NAME["estimate_expense"].invoke(
            {"city": "上海", "days": 3, "hotel_price_per_night": 800, "transport_cost": 1200}
        )
    )
    # 上海上限 600，单价 800 → 必须给出超标提示
    assert exp["hotel_cap_per_night"] == 600
    assert exp["warning"] is not None
    assert exp["total"] == exp["hotel_total"] + exp["meal_total"] + exp["transport_cost"]


def test_schema_validation_rejects_bad_args():
    """pydantic args_schema 在工具执行前拦下脏参数。"""
    flights = TOOLS_BY_NAME["search_flights"]
    # 日期不是 YYYY-MM-DD → DateStr 的校验器拒绝
    with pytest.raises(Exception):
        flights.invoke({"origin": "北京", "destination": "上海", "date": "明天"})
    # nights 超出 [1,30] → 数值边界校验拒绝
    with pytest.raises(Exception):
        TOOLS_BY_NAME["search_hotels"].invoke(
            {"city": "上海", "checkin_date": "2026-06-24", "nights": 0}
        )
    # days 越界
    with pytest.raises(Exception):
        TOOLS_BY_NAME["estimate_expense"].invoke(
            {"city": "上海", "days": 99, "hotel_price_per_night": 500, "transport_cost": 100}
        )


async def test_run_tool_call_happy_path():
    call = {
        "name": "get_weather",
        "args": {"city": "上海", "date": "2026-06-24"},
        "id": "call_1",
    }
    msg = await _run_tool_call(call, TOOLS_BY_NAME)
    assert isinstance(msg, ToolMessage)
    assert msg.tool_call_id == "call_1"
    assert msg.name == "get_weather"
    assert "上海" in msg.content


async def test_run_tool_call_unknown_tool_does_not_raise():
    call = {"name": "search_spaceship", "args": {}, "id": "call_x"}
    msg = await _run_tool_call(call, TOOLS_BY_NAME)
    assert msg.tool_call_id == "call_x"
    assert "未知工具" in msg.content


async def test_run_tool_call_bad_args_feeds_back_for_self_correction():
    """参数校验失败时，不抛异常，而是把错误包成 ToolMessage 回灌（自纠闭环）。"""
    call = {
        "name": "search_flights",
        "args": {"origin": "北京", "destination": "上海", "date": "明天"},
        "id": "call_2",
    }
    msg = await _run_tool_call(call, TOOLS_BY_NAME)
    assert msg.tool_call_id == "call_2"
    assert "失败" in msg.content  # 含"请检查并修正参数后重试"


async def test_gather_preserves_order_and_isolates_failure():
    """并发执行：顺序与 tool_call_id 一一对应；其中一个失败不连累其它。"""
    calls = [
        {
            "name": "search_flights",
            "args": {"origin": "北京", "destination": "上海", "date": "2026-06-24"},
            "id": "a",
        },
        {"name": "nonexistent", "args": {}, "id": "b"},
        {"name": "get_weather", "args": {"city": "上海", "date": "2026-06-24"}, "id": "c"},
    ]
    msgs = await asyncio.gather(*(_run_tool_call(c, TOOLS_BY_NAME) for c in calls))
    assert [m.tool_call_id for m in msgs] == ["a", "b", "c"]  # 顺序保持
    assert "未知工具" in msgs[1].content  # 中间那个失败被隔离
    assert "上海" in msgs[2].content  # 第三个照常成功
