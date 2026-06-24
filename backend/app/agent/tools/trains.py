"""高铁/动车查询工具（确定性 mock）。

和 flights 并列存在，是为了演示「同一意图、多个候选工具」时模型如何按 docstring
选工具（M2 暂不做工具检索/路由，那是之前讨论过的 bind_tools 优化，留作后续话题）。
"""

from __future__ import annotations

import json

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.agent.tools._mock import DateStr, seeded_int, simulate_latency


class TrainQuery(BaseModel):
    origin: str = Field(min_length=1, description='出发城市中文名，如 "北京"')
    destination: str = Field(min_length=1, description='到达城市中文名，如 "上海"')
    date: DateStr = Field(description="出发日期，格式 YYYY-MM-DD（相对日期请先换算）")


@tool(args_schema=TrainQuery)
def search_trains(origin: str, destination: str, date: str) -> str:
    """查询两地之间的高铁/动车车次。

    适用场景：用户想坐高铁/火车、或在机票与高铁间比较时调用。
    返回：车次列表 JSON，每条含车次号、起降时间、历时、二等座/一等座价格(元)、余票；
    按出发时间升序。
    """
    simulate_latency()
    seed = seeded_int(origin, destination, date)
    trains = []
    for i in range(3):
        train_no = f"G{100 + (seed + i * 13) % 800}"
        dep_hour = (seed // (i + 1)) % 12 + 7  # 7~18 点
        duration = (seed % 4) + 4  # 4~7 小时
        second_class = 400 + ((seed >> i) % 10) * 50  # 400~850
        trains.append(
            {
                "train_no": train_no,
                "depart": f"{date} {dep_hour:02d}:{(seed % 6) * 10:02d}",
                "arrive": f"{date} {(dep_hour + duration) % 24:02d}:{(seed % 6) * 10:02d}",
                "duration_h": duration,
                "second_class_price": second_class,
                "first_class_price": int(second_class * 1.6),
                "seats_left": (seed + i) % 30 + 1,
            }
        )
    trains.sort(key=lambda t: t["depart"])
    return json.dumps(
        {"origin": origin, "destination": destination, "date": date, "trains": trains},
        ensure_ascii=False,
    )
