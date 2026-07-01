"""差旅工具集（M2 六件套 + M7 高危两件）。

flights / hotels / trains / weather —— 实时信息查询，确定性 mock；
policy            —— 政策问答（M2 内置规则，M5 换 RAG）；
expense           —— 费用试算（确定性计算）；
book_trip / cancel_booking —— 【M7 高危】下单/取消，需 booking:write 权限 + 人工确认 + 幂等。

只读查询与高危动作放在同一 ALL_TOOLS 里，但由 security/authz 的名单区分对待
（是否需 scope、是否需 HITL）——工具集合"平"，安全策略"分层"。

ALL_TOOLS 给图绑定（bind_tools）用；TOOLS_BY_NAME 给 tools 节点按名查找执行。
"""

from __future__ import annotations

from app.agent.tools.booking import book_trip, cancel_booking
from app.agent.tools.expense import estimate_expense
from app.agent.tools.flights import search_flights
from app.agent.tools.hotels import search_hotels
from app.agent.tools.policy import query_travel_policy
from app.agent.tools.trains import search_trains
from app.agent.tools.weather import get_weather

ALL_TOOLS = [
    search_flights,
    search_hotels,
    search_trains,
    get_weather,
    query_travel_policy,
    estimate_expense,
    book_trip,
    cancel_booking,
]
TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}
