"""航班查询工具（确定性 mock 数据）。

工具设计三要点（面试常考）：
  1. @tool 装饰器把普通函数变成 LLM 可调用的工具；
  2. **docstring 是工具的"说明书"**——模型靠它判断"什么时候该调我"，写不好就乱调；
  3. M2 起用 pydantic 的 args_schema 显式声明参数 + 校验：每个字段的 description
     喂给模型当"参数说明"，类型/格式校验则在工具执行前把脏参数挡掉（校验失败会被
     tools 节点回灌给模型触发自纠）。
"""

from __future__ import annotations

import json

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.agent.tools._mock import DateStr, seeded_int, simulate_latency

# 几家航司，用于生成 mock 航班
_AIRLINES = [
    ("国航", "CA"),
    ("东航", "MU"),
    ("南航", "CZ"),
    ("海航", "HU"),
]


class FlightQuery(BaseModel):
    """search_flights 的参数 schema。Field.description 就是给模型看的参数说明。"""

    origin: str = Field(min_length=1, description='出发城市中文名，如 "北京"')
    destination: str = Field(min_length=1, description='到达城市中文名，如 "上海"')
    date: DateStr = Field(
        description="出发日期，格式 YYYY-MM-DD。相对日期（明天/后天）请先按系统提示里的"
        "今天日期换算成具体日期再传入。"
    )


@tool(args_schema=FlightQuery)
def search_flights(origin: str, destination: str, date: str) -> str:
    """查询指定日期、从出发城市到到达城市的航班。

    适用场景：用户想订机票 / 查航班 / 比价时调用。
    返回：航班列表 JSON，每条含航司、航班号、起降时间、价格(元)、余票，按价格升序。
    """
    simulate_latency()
    seed = seeded_int(origin, destination, date)
    flights = []
    for i in range(3):  # 给 3 个备选
        airline_name, airline_code = _AIRLINES[(seed + i) % len(_AIRLINES)]
        dep_hour = (seed // (i + 1)) % 14 + 6  # 6~19 点起飞
        duration = (seed % 3) + 2  # 2~4 小时
        price = 600 + ((seed >> i) % 18) * 100  # 600~2300
        flights.append(
            {
                "airline": airline_name,
                "flight_no": f"{airline_code}{1000 + (seed + i * 37) % 8999}",
                "depart": f"{date} {dep_hour:02d}:{(seed % 6) * 10:02d}",
                "arrive": f"{date} {(dep_hour + duration) % 24:02d}:{(seed % 6) * 10:02d}",
                "price": price,
                "seats_left": (seed + i) % 9 + 1,
            }
        )
    flights.sort(key=lambda f: f["price"])  # 按价格升序，方便模型推荐最便宜
    return json.dumps(
        {"origin": origin, "destination": destination, "date": date, "flights": flights},
        ensure_ascii=False,
    )
