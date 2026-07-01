"""M9b 刷新令牌轮换测试。

两层：
  1. hermetic（无基建）：access / refresh 令牌类型不可混用；refresh 载荷携带 family+身份可还原。
  2. 集成（需 Postgres + Redis，未起则跳过）：登录发双令牌；刷新旋转（旧的失效）；
     旧令牌重放触发盗用检测并作废整条会话；登出撤销会话。
"""

from __future__ import annotations

from uuid import uuid4

import httpx
import pytest
from jwt import PyJWTError
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import settings
from app.core.security import (
    SCOPE_CHAT,
    Identity,
    create_access_token,
    create_refresh_token,
    decode_access_token,
    decode_refresh_token,
    hash_password,
    identity_from_payload,
)
from app.db.models import Tenant, User
from app.main import app

_ALICE = ("acme", "Acme 差旅", "alice@acme.com", "alice-pass")


# ============================ hermetic：令牌类型 ============================


def test_access_and_refresh_tokens_not_interchangeable():
    ident = Identity(tenant_id=uuid4(), user_id=uuid4(), email="a@b.com", scopes=("chat:write",))
    access = create_access_token(ident)
    refresh = create_refresh_token(ident, family="fam1", jti="jti1")
    # 类型混淆防御：谁都不能冒充对方
    with pytest.raises(PyJWTError):
        decode_access_token(refresh)
    with pytest.raises(PyJWTError):
        decode_refresh_token(access)


def test_refresh_token_carries_family_and_identity():
    tid, uid = uuid4(), uuid4()
    ident = Identity(
        tenant_id=tid, user_id=uid, email="a@b.com", scopes=("chat:write", "booking:write")
    )
    payload = decode_refresh_token(create_refresh_token(ident, family="famX", jti="jtiY"))
    assert payload["family"] == "famX"
    assert payload["jti"] == "jtiY"
    back = identity_from_payload(payload)
    assert back.tenant_id == tid
    assert back.user_id == uid
    assert back.scopes == ("chat:write", "booking:write")
    # access 令牌仍能正常还原身份
    assert decode_access_token(create_access_token(ident)).user_id == uid


# ============================ 集成：刷新流程 ============================


async def _seed_or_skip() -> None:
    """确保 alice 演示账号存在；连不上 Postgres 则跳过集成测试。"""
    try:
        engine = create_async_engine(settings.database_url)
        sm = async_sessionmaker(engine, expire_on_commit=False)
        async with sm() as s:
            slug, name, email, pw = _ALICE
            tenant = (
                await s.execute(select(Tenant).where(Tenant.slug == slug))
            ).scalar_one_or_none()
            if tenant is None:
                tenant = Tenant(name=name, slug=slug)
                s.add(tenant)
                await s.flush()
            user = (
                await s.execute(
                    select(User).where(User.tenant_id == tenant.id, User.email == email)
                )
            ).scalar_one_or_none()
            if user is None:
                s.add(
                    User(
                        tenant_id=tenant.id,
                        email=email,
                        password_hash=hash_password(pw),
                        display_name=email,
                        scopes=[SCOPE_CHAT],
                    )
                )
            await s.commit()
        await engine.dispose()
    except (OSError, SQLAlchemyError) as e:  # noqa: BLE001
        pytest.skip(f"需要 Postgres（make up）：{e!r}")


async def _login(client: httpx.AsyncClient) -> dict:
    r = await client.post(
        "/auth/login",
        json={"tenant_slug": "acme", "email": "alice@acme.com", "password": "alice-pass"},
    )
    assert r.status_code == 200, r.text
    return r.json()


async def test_login_issues_token_pair():
    await _seed_or_skip()
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            d = await _login(client)
            assert d["access_token"] and d["refresh_token"]
            assert d["token_type"] == "bearer"
            assert d["expires_in"] > 0


async def test_refresh_rotates_and_detects_reuse():
    await _seed_or_skip()
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            rt1 = (await _login(client))["refresh_token"]

            # 用 rt1 刷新 → 拿到 rt2/at2（旋转：rt2 != rt1）
            r2 = await client.post("/auth/refresh", json={"refresh_token": rt1})
            assert r2.status_code == 200, r2.text
            rt2, at2 = r2.json()["refresh_token"], r2.json()["access_token"]
            assert rt2 != rt1

            # 新 access 可用
            ok = await client.get("/conversations", headers={"Authorization": f"Bearer {at2}"})
            assert ok.status_code == 200, ok.text

            # 旧 rt1 再用 → 盗用检测 → 401
            assert (
                await client.post("/auth/refresh", json={"refresh_token": rt1})
            ).status_code == 401

            # 盗用检测已作废整条会话 → 连合法的 rt2 也失效
            assert (
                await client.post("/auth/refresh", json={"refresh_token": rt2})
            ).status_code == 401


async def test_logout_revokes_session():
    await _seed_or_skip()
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            rt = (await _login(client))["refresh_token"]
            assert (
                await client.post("/auth/logout", json={"refresh_token": rt})
            ).status_code == 204
            # 登出后该刷新令牌失效
            assert (
                await client.post("/auth/refresh", json={"refresh_token": rt})
            ).status_code == 401
