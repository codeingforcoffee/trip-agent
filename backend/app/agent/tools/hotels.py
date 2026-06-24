"""酒店查询工具（确定性 mock）。

校验看点：nights 用 Field(ge=1, le=30) 做**数值边界**校验——模型若传 0 或 999，
pydantic 直接拒绝，错误回灌触发自纠。
"""

from __future__ import annotations

import json

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.agent.tools._mock import DateStr, seeded_int, simulate_latency

# 品牌 -> 星级，用于生成 mock 酒店
_BRANDS = [
    ("汉庭", 3),
    ("如家", 3),
    ("全季", 4),
    ("亚朵", 4),
    ("希尔顿", 5),
]


class HotelQuery(BaseModel):
    city: str = Field(min_length=1, description='城市中文名，如 "上海"')
    checkin_date: DateStr = Field(description="入住日期，格式 YYYY-MM-DD")
    nights: int = Field(default=1, ge=1, le=30, description="入住晚数，1~30，默认 1")


@tool(args_schema=HotelQuery)
def search_hotels(city: str, checkin_date: str, nights: int = 1) -> str:
    """查询指定城市、入住日期的酒店，返回按性价比（每晚价格）升序的备选。

    适用场景：用户要订酒店 / 查住宿 / 问哪家划算时调用。
    返回：酒店列表 JSON，每条含名称、星级、每晚价格、评分、距市中心距离、总价。
    """
    simulate_latency()
    seed = seeded_int(city, checkin_date)
    hotels = []
    for i in range(3):
        name, star = _BRANDS[(seed + i) % len(_BRANDS)]
        per_night = 200 + ((seed >> i) % 12) * 80  # 200~1080
        rating = round(7.0 + ((seed + i) % 30) / 10, 1)  # 7.0~9.9
        distance_km = round(((seed >> (i + 1)) % 80) / 10, 1)  # 0~8.0 km
        hotels.append(
            {
                "name": f"{city}{name}酒店",
                "star": star,
                "price_per_night": per_night,
                "rating": rating,
                "distance_km": distance_km,
                "total_price": per_night * nights,
            }
        )
    hotels.sort(key=lambda h: h["price_per_night"])
    return json.dumps(
        {"city": city, "checkin_date": checkin_date, "nights": nights, "hotels": hotels},
        ensure_ascii=False,
    )
