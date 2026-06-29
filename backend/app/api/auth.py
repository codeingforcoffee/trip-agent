"""登录与令牌签发（M3）。

登录是个有意思的边界情形：**此刻还没有身份，却要在 RLS 下查 users**。
解法顺序：
  1. 按 tenant_slug 查 tenants（该表无 RLS，普通角色可读）→ 拿到 tenant_id；
  2. set_tenant_context 设好租户上下文；
  3. 再按 email 查 user（此时 RLS 生效，只能查到本租户的用户）；
  4. 校验密码 → 签发 JWT。

错误信息刻意【不区分】"租户不存在 / 用户不存在 / 密码错"，统一返回 401，
避免账号枚举（攻击者无法通过报错差异探测哪些账号存在）。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import Identity, create_access_token, verify_password
from app.db.models import Tenant, User
from app.db.session import get_session, set_tenant_context

router = APIRouter(prefix="/auth", tags=["auth"])

_LOGIN_FAILED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED, detail="租户、账号或密码不正确"
)


class LoginRequest(BaseModel):
    tenant_slug: str
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest, session: AsyncSession = Depends(get_session)) -> TokenResponse:
    tenant = (
        await session.execute(select(Tenant).where(Tenant.slug == body.tenant_slug))
    ).scalar_one_or_none()
    if tenant is None:
        raise _LOGIN_FAILED

    # 设好租户上下文后再查 user —— 之后这条会话的查询都受 RLS 按本租户过滤
    await set_tenant_context(session, tenant.id)
    user = (
        await session.execute(select(User).where(User.email == body.email))
    ).scalar_one_or_none()
    if user is None or not verify_password(body.password, user.password_hash):
        raise _LOGIN_FAILED

    identity = Identity(
        tenant_id=tenant.id,
        user_id=user.id,
        email=user.email,
        scopes=tuple(user.scopes),
    )
    return TokenResponse(access_token=create_access_token(identity))
