"""审计日志（M7）：把安全相关事件落库，做到"谁、在哪个租户、做了什么"可追溯、不可抵赖。

写哪些事件（安全合规的最小集）：
  - tool.denied   —— 越权尝试被拒（无 scope）。这是入侵检测的信号源。
  - hitl.rejected —— 高危动作被用户拒绝。
  - tool.executed —— 高危动作确实执行了（下单/取消），事后可对账、可举证。
  （M7b 再加 injection.detected / pii.masked。）

存储：复用 M3 的 audit_logs 表（JSONB detail 存结构化上下文，便于按字段查询）。
连接用 **app 角色（受 RLS）**、事务级设租户上下文——和 M6b 记忆落库同一套安全姿势：
即便审计代码有 bug，也绝写不进/读不到别的租户的审计。

铁律：**审计失败绝不能阻断主流程**（记不上日志不该让用户下不了单，也不该让越权者
因为审计报错反而"漏过"拦截——拦截在前，审计在后）。所以调用方一律 try/except 包裹，
这里失败也只吞进日志。
"""

from __future__ import annotations

import uuid
from functools import lru_cache

from app.core.logging import get_logger
from app.db.models import AuditLog
from app.db.session import set_tenant_context

log = get_logger("app.security.audit")


@lru_cache(maxsize=1)
def _sessionmaker():
    """审计用的会话工厂（懒建单例，app 角色受 RLS）。

    与 M6b 记忆的 _sessionmaker 同构：给无 app.state 的图/CLI 用；测试用 cache_clear() 重置。
    """
    from app.db.session import build_engine, build_sessionmaker

    return build_sessionmaker(build_engine())


async def record_audit(
    tenant_id: str, user_id: str | None, action: str, detail: dict | None = None
) -> None:
    """写一条审计。事务级设租户上下文（RLS）→ INSERT → commit，全在同一事务。

    user_id 可为空（系统动作）。detail 是任意结构化上下文（工具名/参数/所需 scope 等）。
    """
    async with _sessionmaker()() as s:
        await set_tenant_context(s, tenant_id)  # 事务级 RLS 上下文；commit 后自动失效
        s.add(
            AuditLog(
                tenant_id=uuid.UUID(str(tenant_id)),
                user_id=uuid.UUID(str(user_id)) if user_id else None,
                action=action,
                detail=detail or {},
            )
        )
        await s.commit()
        log.info("audit.record", action=action, **(detail or {}))
