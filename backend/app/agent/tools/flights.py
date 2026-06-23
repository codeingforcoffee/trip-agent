"""航班查询工具（M1 用确定性 mock 数据）。

工具设计的三个要点（面试常考）：
  1. 用 @tool 装饰器把普通函数变成 LLM 可调用的工具；
  2. 函数签名的类型注解 → 自动生成 JSON Schema 给模型，模型据此决定传什么参数；
  3. **docstring 极其重要**——它就是工具的"说明书"，模型靠它判断"什么时候该调我、
     每个参数填什么"。写不好，模型就乱调或不调。

为什么用"确定性 mock"而非真随机？
  - 同样的输入永远返回同样的结果 → 后续 M8 的离线评测可复现（这是"可离线评测"的地基）；
  - 真实差旅 API（携程/航司）接入是工程问题，不是 Agent 核心，留到需要时再换实现。
"""

from __future__ import annotations

import hashlib
import json

from langchain_core.tools import tool

# 几家航司，用于生成 mock 航班
_AIRLINES = [
    ("国航", "CA"),
    ("东航", "MU"),
    ("南航", "CZ"),
    ("海航", "HU"),
]


def _seeded_int(*parts: str) -> int:
    """用输入拼一个稳定哈希 → 确定性'伪随机'，保证同输入同输出。"""
    h = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return int(h[:8], 16)


@tool
def search_flights(origin: str, destination: str, date: str) -> str:
    """查询指定日期、从出发城市到到达城市的航班。

    适用场景：用户想订机票 / 查航班 / 比价时调用。

    参数：
        origin: 出发城市中文名，如 "北京"。
        destination: 到达城市中文名，如 "上海"。
        date: 出发日期，格式 YYYY-MM-DD（如 "2026-06-24"）。相对日期（明天/后天）
              请先根据系统提示里的"今天日期"换算成具体日期再传入。

    返回：航班列表的 JSON 字符串，每条含航司、航班号、起降时间、价格(元)、余票。
    """
    seed = _seeded_int(origin, destination, date)
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
