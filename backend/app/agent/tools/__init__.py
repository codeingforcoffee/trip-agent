"""差旅工具集。

M1 只有一个 search_flights；M2 会加 hotels/trains/weather/policy/expense，
并把"执行工具"的节点升级为并发执行。

ALL_TOOLS 是给图绑定用的工具清单；TOOLS_BY_NAME 给 tools 节点按名查找。
"""

from __future__ import annotations

from app.agent.tools.flights import search_flights

ALL_TOOLS = [search_flights]
TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}
