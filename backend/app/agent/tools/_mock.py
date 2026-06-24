"""工具层共用基建（M2）。

为什么所有工具都用「确定性 mock」而非真随机？
  - 同输入永远同输出 → M8 的离线评测可复现（这是"可离线评测"的地基）；
  - 真实差旅 API（携程/航司/天气）接入是工程问题，不是 Agent 核心，留到需要时再换实现。

这里集中三样东西，避免六个工具各写一份：
  1. seeded_int      —— 用输入拼稳定哈希，做确定性「伪随机」；
  2. simulate_latency —— 可选的模拟网络延迟，用来在演示里「看见」并发收益；
  3. DateStr         —— 复用的「YYYY-MM-DD 字符串」类型，自带 pydantic 校验。
"""

from __future__ import annotations

import hashlib
import time
from datetime import date as _date
from typing import Annotated

from pydantic import AfterValidator

from app.core.config import settings


def seeded_int(*parts: str) -> int:
    """把任意输入拼成一个稳定的 32-bit 整数（确定性'伪随机'种子）。

    同样的 parts 永远得到同样的数 → 工具输出可复现。用 sha256 是图省事，
    要的不是密码学强度，而是「输入变一点、输出就大变」的散列特性。
    """
    h = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return int(h[:8], 16)


def simulate_latency() -> None:
    """按配置 sleep，模拟一次外部 API 往返。默认 0（测试/评测保持瞬时、时间确定）。

    关键细节（面试会问"为什么同步 sleep 不会卡死事件循环"）：
      工具是**同步**函数，LangChain 的 tool.ainvoke 会把它丢进**线程池**执行；
      time.sleep 在 sleep 期间**释放 GIL**，于是 asyncio.gather 同时发起的多个
      工具能在各自线程里真正并行 sleep —— 这正是「3 个工具并发只花 ~1 倍延迟
      而非 3 倍」的底层原因。
    """
    ms = settings.tool_mock_latency_ms
    if ms > 0:
        time.sleep(ms / 1000)


def _validate_date(v: str) -> str:
    """校验 YYYY-MM-DD。失败抛 ValueError —— 这条错误会被 tools 节点捕获、
    包成 ToolMessage 回灌给模型，触发它把"明天"换算成具体日期后**自我纠正**重试。
    """
    try:
        _date.fromisoformat(v)
    except ValueError as e:
        raise ValueError(f"date 必须是 YYYY-MM-DD 格式（你传的是 {v!r}）") from e
    return v


# 复用类型：哪个工具需要日期参数，字段类型写成 DateStr 即可自动带上校验。
# Annotated[str, AfterValidator(...)] 是 pydantic v2 的惯用法：先当 str 解析，
# 再跑我们的校验函数。比每个 schema 重复写 @field_validator 干净得多。
DateStr = Annotated[str, AfterValidator(_validate_date)]
