"""天气查询工具（确定性 mock）。

这是 fan-out 演示里的"第三个工具"：用户问「查机票、酒店和天气」时，模型一轮吐出
flights/hotels/weather 三个 tool_call，tools 节点用 asyncio.gather 并发执行。
"""

from __future__ import annotations

import json

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.agent.tools._mock import DateStr, seeded_int, simulate_latency

_CONDITIONS = ["晴", "多云", "阴", "小雨", "雷阵雨", "雪"]
# 天气 -> 给差旅者的一句建议
_ADVICE = {
    "晴": "紫外线强，备防晒。",
    "多云": "适宜出行。",
    "阴": "适宜出行，注意保暖。",
    "小雨": "记得带伞。",
    "雷阵雨": "航班可能延误，预留缓冲时间。",
    "雪": "路面湿滑，预留充足赶车时间。",
}


class WeatherQuery(BaseModel):
    city: str = Field(min_length=1, description='城市中文名，如 "上海"')
    date: DateStr = Field(description="查询日期，格式 YYYY-MM-DD（相对日期请先换算）")


@tool(args_schema=WeatherQuery)
def get_weather(city: str, date: str) -> str:
    """查询某城市某天的天气预报。

    适用场景：用户出行前问天气、或需根据天气给出行建议时调用。
    返回：JSON，含天气状况、最高/最低温(℃)、降水概率(%)、出行建议。
    """
    simulate_latency()
    seed = seeded_int(city, date)
    condition = _CONDITIONS[seed % len(_CONDITIONS)]
    low = seed % 15 + 3  # 3~17 ℃
    high = low + (seed % 10) + 4  # 比最低高 4~13
    precip = (seed % 10) * 10  # 0~90 %
    return json.dumps(
        {
            "city": city,
            "date": date,
            "condition": condition,
            "temp_low": low,
            "temp_high": high,
            "precip_prob": precip,
            "advice": _ADVICE[condition],
        },
        ensure_ascii=False,
    )
