"""高危差旅工具（M7）：下单 book_trip / 取消 cancel_booking。

这是 M7 授权/HITL/幂等/审计几条线的**落地对象**——前面 M0~M6 全是只读查询，
没有一个会花钱、不可逆的动作，授权与人在环路无从演示。这里补上。

三层防护叠加，各解决不同问题（面试要能分清）：
  1. **scope 授权**（security/authz + graph 工具层）：没有 booking:write 的用户根本调不动。
  2. **HITL 人工确认**（graph confirm 节点 + interrupt）：防"未经同意就执行"。
  3. **幂等 key + M4 锁**（本文件）：防"同一意图被重复执行"（网络重试/用户重发/图重放）
     导致重复扣款；锁防"两个并发请求给同一意图双开单"。三者缺一，高危动作就有漏洞。

关键：非幂等工具**绝不自动重试**（重试=重复扣款），import 时 mark_non_idempotent 登记，
M4 的 call_tool_resilient 便对它们只试一次。租户身份只从 config.configurable 取，
**绝不信任 LLM 在参数里传的 tenant_id**（否则 A 租户能诱导 agent 给 B 租户下单）。
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.agent.reliability import mark_non_idempotent
from app.core.config import settings
from app.core.logging import get_logger
from app.infra.locks import RedisLock, lock_key
from app.infra.redis_client import get_redis_client

log = get_logger("app.agent.tools.booking")


def _idem_key(tenant_id: str, user_id: str | None, *parts: object) -> str:
    """由(租户+用户+行程要素)派生的稳定幂等键。

    同一"下单意图"→ 同一 key → 同一订单号（见下）→ 天然去重。用 sha1 取前 16 位：
    够低碰撞、又不把租户/用户明文写进 Redis key。
    """
    raw = ":".join(str(p) for p in (tenant_id, user_id, *parts))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _result_key(tenant_id: str, idem: str) -> str:
    """幂等结果的 Redis key（带租户前缀，缓存层隔离，与 M3/M4 命名一致）。"""
    return f"t:{tenant_id}:book:result:{idem}"


class BookTripInput(BaseModel):
    trip_type: Literal["flight", "train", "hotel"] = Field(
        description="预订类型：flight/train/hotel"
    )
    item_id: str = Field(
        min_length=1, description="要预订的具体项，如航班号 CA1831 / 车次 G1 / 酒店标识"
    )
    date: str = Field(description="出行或入住日期，YYYY-MM-DD")
    price: float = Field(ge=0, description="下单金额（元）")


@tool(args_schema=BookTripInput)
async def book_trip(
    trip_type: str, item_id: str, date: str, price: float, config: RunnableConfig
) -> str:
    """【高危·会产生真实订单与扣款】预订一张机票 / 火车票 / 酒店。

    仅在用户**明确要求下单/预订/出票**且信息齐全时调用（查询用 search_* 工具，不要用本工具）。
    调用它需要 booking:write 权限，并会在执行前请用户人工确认。返回订单号与状态（JSON）。
    """
    cfg = config.get("configurable") or {}
    tenant_id = cfg.get("tenant_id")
    user_id = cfg.get("user_id")
    if not tenant_id:
        # fail-closed：拿不到可信身份宁可不下单（身份只认 config，不认 LLM 传参）
        return "下单失败：缺少可信的租户身份，已拒绝（fail-closed）。"

    idem = _idem_key(str(tenant_id), user_id, trip_type, item_id, date)
    # 订单号也由幂等键派生 → 同一意图即使缓存过期后重算，订单号仍一致（可对账）
    order_id = "TRIP-" + hashlib.sha1(idem.encode("utf-8")).hexdigest()[:8].upper()
    redis = get_redis_client()
    rkey = _result_key(str(tenant_id), idem)

    # M4 锁：把"查是否已下单 → 创建 → 落幂等结果"围成临界区。
    # 没有锁时两个并发请求可能同时读到"未下单"、各自创建 → 双开单；锁让它们串行，
    # 第二个进临界区时已能读到第一个写下的结果 → 命中幂等分支。auto_renew 防临界区偏长时锁过期。
    async with RedisLock(redis, lock_key(str(tenant_id), f"book:{idem}"), auto_renew=True):
        cached = await redis.get(rkey)
        if cached:
            log.info("booking.idempotent_hit", order_id=order_id, item_id=item_id)
            return cached  # 幂等重放：同一意图已下过单 → 原样返回，绝不重复扣款

        result = json.dumps(
            {
                "status": "confirmed",
                "order_id": order_id,
                "trip_type": trip_type,
                "item_id": item_id,
                "date": date,
                "price": price,
                "note": "已为你完成预订（演示环境：不真实出票/扣款）。",
            },
            ensure_ascii=False,
        )
        await redis.set(rkey, result, ex=settings.booking_idem_ttl_s)
        log.info("booking.created", order_id=order_id, item_id=item_id, price=price)
        return result


class CancelBookingInput(BaseModel):
    order_id: str = Field(min_length=1, description="要取消的订单号，如 TRIP-XXXXXXXX")


@tool(args_schema=CancelBookingInput)
async def cancel_booking(order_id: str, config: RunnableConfig) -> str:
    """【高危·不可逆】取消一张已存在的订单。需 booking:write 权限并经人工确认。"""
    cfg = config.get("configurable") or {}
    tenant_id = cfg.get("tenant_id")
    if not tenant_id:
        return "取消失败：缺少可信的租户身份，已拒绝（fail-closed）。"

    redis = get_redis_client()
    # 取消**天然幂等**（取消两次仍是已取消），不需缓存"首次结果"；但仍用锁避免与下单/并发取消竞态。
    async with RedisLock(redis, lock_key(str(tenant_id), f"cancel:{order_id}"), auto_renew=True):
        await redis.set(
            f"t:{tenant_id}:book:cancelled:{order_id}", "1", ex=settings.booking_idem_ttl_s
        )
        log.info("booking.cancelled", order_id=order_id)
        return json.dumps(
            {"status": "cancelled", "order_id": order_id, "note": "订单已取消（演示环境）。"},
            ensure_ascii=False,
        )


# 非幂等：登记进 M4 可靠性层的名单，使其**不自动重试**（重试=重复下单/取消）。
mark_non_idempotent("book_trip")
mark_non_idempotent("cancel_booking")
