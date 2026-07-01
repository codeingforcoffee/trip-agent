"""登录 / 刷新 / 登出（M3 登录 + M9b 刷新令牌轮换）。

登录是个有意思的边界情形：**此刻还没有身份，却要在 RLS 下查 users**。
解法顺序：
  1. 按 tenant_slug 查 tenants（该表无 RLS，普通角色可读）→ 拿到 tenant_id；
  2. set_tenant_context 设好租户上下文；
  3. 再按 email 查 user（此时 RLS 生效，只能查到本租户的用户）；
  4. 校验密码 → 签发 access + refresh 令牌。

错误信息刻意【不区分】"租户不存在 / 用户不存在 / 密码错"，统一返回 401，避免账号枚举。

M9b 刷新令牌轮换（refresh token rotation）：
  - 登录发【短命 access + 长命 refresh】；access 过期后用 refresh 换新，不必重新登录；
  - 每次刷新都【旋转】：换新 refresh、旧的立即失效（一次性）；
  - **盗用检测**：旧 refresh 被重放 → 判定令牌被盗 → 作废整条会话（见 infra/refresh_store.py）。
"""

from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, status
from jwt import PyJWTError
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.core.security import (
    Identity,
    create_access_token,
    create_refresh_token,
    decode_refresh_token,
    identity_from_payload,
    verify_password,
)
from app.db.models import Tenant, User
from app.db.session import get_session, set_tenant_context
from app.infra.refresh_store import RefreshStore

log = get_logger("app.api.auth")

router = APIRouter(prefix="/auth", tags=["auth"])

_LOGIN_FAILED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED, detail="租户、账号或密码不正确"
)
_REFRESH_FAILED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="刷新令牌无效或已过期，请重新登录",
    headers={"WWW-Authenticate": "Bearer"},
)


class LoginRequest(BaseModel):
    tenant_slug: str
    email: str
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # access 令牌有效秒数，供前端在过期前主动刷新


def _issue_tokens(identity: Identity, family: str) -> tuple[TokenResponse, str]:
    """为某身份签发一对令牌，返回 (令牌响应, 本次 jti)。jti 由调用方登记到服务端家族记录。"""
    jti = uuid4().hex
    access = create_access_token(identity)
    refresh = create_refresh_token(identity, family=family, jti=jti)
    return (
        TokenResponse(
            access_token=access,
            refresh_token=refresh,
            expires_in=settings.jwt_expire_minutes * 60,
        ),
        jti,
    )


@router.post("/login", response_model=TokenResponse)
async def login(
    body: LoginRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> TokenResponse:
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
    store = RefreshStore(request.app.state.redis)
    family = uuid4().hex  # 新会话 = 新家族
    tokens, jti = _issue_tokens(identity, family)
    await store.register(str(identity.tenant_id), family, jti)
    log.info("auth.login", tenant=str(tenant.id), user=str(user.id), family=family)
    return tokens


@router.post("/refresh", response_model=TokenResponse)
async def refresh(body: RefreshRequest, request: Request) -> TokenResponse:
    """用 refresh 令牌换一对新令牌（旋转）。旧 refresh 立即失效；检测到重放则作废整条会话。"""
    try:
        payload = decode_refresh_token(body.refresh_token)
    except PyJWTError:
        raise _REFRESH_FAILED from None

    family, old_jti = payload.get("family"), payload.get("jti")
    tenant_id = payload.get("tenant")
    if not (family and old_jti and tenant_id):
        raise _REFRESH_FAILED

    identity = identity_from_payload(payload)
    store = RefreshStore(request.app.state.redis)
    new_jti = uuid4().hex
    outcome = await store.rotate(tenant_id, family, old_jti, new_jti)

    if outcome == "reuse":
        # 旧令牌被重放：极可能是令牌被盗，两边各刷一次。家族已被 Lua 一并作废。
        log.warning("auth.refresh_reuse_detected", tenant=tenant_id, family=family)
        raise _REFRESH_FAILED
    if outcome != "ok":  # 'revoked'：家族不存在（已登出/过期）
        raise _REFRESH_FAILED

    access = create_access_token(identity)
    new_refresh = create_refresh_token(identity, family=family, jti=new_jti)
    log.info("auth.refresh", tenant=tenant_id, family=family)
    return TokenResponse(
        access_token=access,
        refresh_token=new_refresh,
        expires_in=settings.jwt_expire_minutes * 60,
    )


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(body: RefreshRequest, request: Request) -> None:
    """登出：撤销该刷新令牌所属的整条会话家族。无效令牌静默成功（幂等，不泄漏信息）。"""
    try:
        payload = decode_refresh_token(body.refresh_token)
    except PyJWTError:
        return  # 已经不是有效令牌，等价于已登出
    family, tenant_id = payload.get("family"), payload.get("tenant")
    if family and tenant_id:
        await RefreshStore(request.app.state.redis).revoke(tenant_id, family)
        log.info("auth.logout", tenant=tenant_id, family=family)
