"""工具授权（M7）：把「谁能调哪个工具」做成**确定性**的 scope 校验。

为什么授权必须在**工具层**做、而不是写进 system prompt（面试核心红线）：
  - Agent 是"混淆的代理人（confused deputy）"——它握有很高的后端权限（能调所有工具），
    却代表一个可能低权限的用户行事。若不在工具执行前校验用户 scope，用户就能用自然语言
    诱导 agent 替他做他本无权做的事（下单、取消、导出）。
  - system prompt 写"没权限别下单"是**概率性**约束，可被 prompt injection 绕过；
    scope 校验是**确定性**代码分支，模型再怎么被诱导也过不了。这条 + M3 的 RLS（数据层）
    才是真正的硬边界，guards（注入/PII 检测）只是提高攻击成本的概率性护栏。

两张表，职责正交：
  - TOOL_REQUIRED_SCOPE：工具 → 所需 scope。未列出的（只读查询类）无需特殊 scope，默认放行。
  - HIGH_RISK_TOOLS：会产生副作用/不可逆动作的工具，执行前需 HITL 人工确认（interrupt）。
    "需要授权"与"高危需确认"是两件事：查询类可能都不需授权；下单既需授权又需确认。
"""

from __future__ import annotations

from collections.abc import Sequence

from app.core.security import SCOPE_BOOKING

# 工具名 → 调用它所需的 scope。只读查询（flights/hotels/trains/weather/policy/expense）
# 不在表内 → 无需特殊权限。新增有副作用的工具时在这里登记它的 required scope。
TOOL_REQUIRED_SCOPE: dict[str, str] = {
    "book_trip": SCOPE_BOOKING,
    "cancel_booking": SCOPE_BOOKING,
}

# 高危工具：会产生真实订单/扣款/不可逆变更 → 执行前必须人工确认（HITL）。
# 用 frozenset：不可变、成员判定 O(1)、语义上是"名单常量"。
HIGH_RISK_TOOLS: frozenset[str] = frozenset({"book_trip", "cancel_booking"})


def required_scope(tool_name: str) -> str | None:
    """该工具需要的 scope；None 表示无需特殊权限（只读工具）。"""
    return TOOL_REQUIRED_SCOPE.get(tool_name)


def is_high_risk(tool_name: str) -> bool:
    """是否高危工具（执行前需 HITL 人工确认）。"""
    return tool_name in HIGH_RISK_TOOLS


def has_required_scope(tool_name: str, scopes: Sequence[str] | None) -> bool:
    """用户是否具备调用该工具所需的 scope。无需授权的工具恒 True（fail-open 仅对只读）。

    注意 fail-**closed** 的语义在调用方：required 存在但 scopes 缺失 → False → 拒绝执行。
    这里只做纯判定，落库审计与拒绝话术在图/工具层。
    """
    needed = required_scope(tool_name)
    return needed is None or needed in (scopes or [])
