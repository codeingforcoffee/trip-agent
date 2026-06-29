"""FastAPI 依赖（M3）：把"请求边界的身份"翻译成"可信的 Identity + 租户态 DB 会话"。

这是身份贯穿的枢纽：
  HTTP Authorization: Bearer <jwt>
    → get_identity（验签解码，失败 401）
      → get_tenant_session（开事务、设 RLS 租户上下文、给路由用）
      → require_scopes（按需校验权限，缺权限 403）

设计原则：**身份在这里【一次性】从签名 token 建立，之后向下游作为不可变上下文流动**，
绝不从请求体/对话内容里反推 —— 这是防越权与防注入的根。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Coroutine
from typing import Any

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import Identity, decode_access_token
from app.db.session import set_tenant_context

# auto_error=False：自己处理缺失/非法，统一返回 401 + WWW-Authenticate（语义更准）。
_bearer = HTTPBearer(auto_error=False)

_UNAUTHORIZED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="未认证或令牌无效",
    headers={"WWW-Authenticate": "Bearer"},
)


async def get_identity(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> Identity:
    """从 Bearer 令牌解析身份。无头 / 验签失败 / 过期 → 401。"""
    if creds is None or not creds.credentials:
        raise _UNAUTHORIZED
    try:
        return decode_access_token(creds.credentials)
    except jwt.PyJWTError:  # 签名错、过期、格式非法都归到这
        raise _UNAUTHORIZED from None


async def get_tenant_session(
    request: Request,
    identity: Identity = Depends(get_identity),
) -> AsyncIterator[AsyncSession]:
    """开一个【已设好租户上下文】的事务会话给路由用。

    顺序很关键：先 set_tenant_context（开启事务并写入 app.current_tenant），
    之后路由里的所有查询都在这条事务内、受 RLS 按本租户过滤。正常退出提交，
    异常回滚；无论哪种，事务结束时租户上下文自动失效，连接干净归还池子。
    """
    sessionmaker = request.app.state.db_sessionmaker
    async with sessionmaker() as session:
        await set_tenant_context(session, identity.tenant_id)
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def require_scopes(
    *required: str,
) -> Callable[[Identity], Coroutine[Any, Any, Identity]]:
    """依赖工厂：要求身份具备指定的全部 scope，否则 403。M7 工具授权会复用思路。

    用法：`identity: Identity = Depends(require_scopes(SCOPE_BOOKING))`
    """

    async def checker(identity: Identity = Depends(get_identity)) -> Identity:
        missing = [s for s in required if not identity.has_scope(s)]
        if missing:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"缺少所需权限: {missing}",
            )
        return identity

    return checker
