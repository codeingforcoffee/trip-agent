"""差旅费用试算工具（确定性计算，不依赖随机）。

校验看点：days 用 ge/le 卡边界，两个金额用 ge=0 防负数——演示「数值类参数」的
约束式校验。它和 policy 工具配合：policy 给规则，expense 把规则套到具体数字上试算。
"""

from __future__ import annotations

import json

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.agent.tools._mock import simulate_latency

# 一线城市住宿上限更高（与 policy 工具里的政策口径一致）
_TIER1 = {"北京", "上海", "广州", "深圳"}
_PER_DIEM = 100  # 每日餐补（元），与 policy 一致


class ExpenseQuery(BaseModel):
    city: str = Field(min_length=1, description="出差目的地城市中文名")
    days: int = Field(ge=1, le=30, description="出差天数，1~30")
    hotel_price_per_night: float = Field(ge=0, description="预订的酒店每晚价格(元)")
    transport_cost: float = Field(ge=0, description="往返交通总价(元，机票或高铁)")


@tool(args_schema=ExpenseQuery)
def estimate_expense(
    city: str, days: int, hotel_price_per_night: float, transport_cost: float
) -> str:
    """试算一次差旅的总费用与报销情况：交通 + 住宿 + 餐补，并对照政策上限给出超标提示。

    适用场景：用户已知行程要素（城市/天数/酒店价/交通费），想估算总花费或问"能不能全报"时调用。
    返回：JSON，含各项金额、合计、住宿单价上限、以及是否超标的提示。
    """
    simulate_latency()
    nights = days  # 简化：出差 N 天即住 N 晚
    hotel_cap = 600 if city in _TIER1 else 450
    hotel_total = round(hotel_price_per_night * nights, 2)
    meal_total = _PER_DIEM * days
    total = round(hotel_total + meal_total + transport_cost, 2)
    over_cap = hotel_price_per_night > hotel_cap
    warning = (
        f"酒店单价 {hotel_price_per_night} 元/晚 超过 {city} 上限 {hotel_cap} 元/晚，"
        f"超出部分（约 {round((hotel_price_per_night - hotel_cap) * nights, 2)} 元）需自理。"
        if over_cap
        else None
    )
    return json.dumps(
        {
            "city": city,
            "days": days,
            "transport_cost": transport_cost,
            "hotel_total": hotel_total,
            "meal_total": meal_total,
            "total": total,
            "hotel_cap_per_night": hotel_cap,
            "warning": warning,
        },
        ensure_ascii=False,
    )
