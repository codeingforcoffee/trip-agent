"""M3 集成测试：多租户隔离（需要真实 Postgres，未起则整文件跳过）。

两层验证：
  1. HTTP 端到端：alice(acme) / bob(globex) 各自登录拿 token，alice 建会话，
     alice 能看到、bob 看不到 —— 全程应用代码【没有一句租户过滤】，隔离靠 RLS。
  2. DB 层直测：用应用角色 trip_app 连库，验证 fail-closed、跨租户不可见、WITH CHECK
     拒绝越权写入 —— 证明就算应用层完全不设防，数据库这道也兜得住。
"""

from __future__ import annotations

from uuid import uuid4

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.exc import DBAPIError, SQLAlchemyError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import settings
from app.core.security import SCOPE_CHAT, hash_password
from app.db.models import Conversation, Tenant, User
from app.db.session import set_tenant_context
from app.main import app

# 演示账号（与 scripts/seed.py 一致）
_ACME = ("acme", "Acme 差旅", "alice@acme.com", "alice-pass")
_GLOBEX = ("globex", "Globex 出行", "bob@globex.com", "bob-pass")


async def _seed_or_skip() -> None:
    """用超级用户连库确保演示数据存在；连不上则跳过整组集成测试。"""
    try:
        engine = create_async_engine(settings.database_url)
        sm = async_sessionmaker(engine, expire_on_commit=False)
        async with sm() as s:
            for slug, name, email, pw in (_ACME, _GLOBEX):
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


# ============================ HTTP 端到端 ============================


async def test_http_cross_tenant_isolation():
    await _seed_or_skip()

    async with app.router.lifespan_context(app):  # 触发 lifespan，建好连接池到 app.state
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # 1) 两个租户分别登录
            ra = await client.post(
                "/auth/login",
                json={"tenant_slug": "acme", "email": "alice@acme.com", "password": "alice-pass"},
            )
            rb = await client.post(
                "/auth/login",
                json={"tenant_slug": "globex", "email": "bob@globex.com", "password": "bob-pass"},
            )
            assert ra.status_code == 200, ra.text
            assert rb.status_code == 200, rb.text
            ha = {"Authorization": f"Bearer {ra.json()['access_token']}"}
            hb = {"Authorization": f"Bearer {rb.json()['access_token']}"}

            # 2) alice 建一条带唯一标题的会话
            title = f"Acme行程-{uuid4()}"
            rc = await client.post("/conversations", json={"title": title}, headers=ha)
            assert rc.status_code == 200, rc.text
            conv = rc.json()
            assert conv["title"] == title

            # 3) alice 看得到自己的会话
            la = await client.get("/conversations", headers=ha)
            assert any(c["id"] == conv["id"] for c in la.json())

            # 4) bob 看不到 alice 的会话（隔离！应用层并没有写任何过滤条件）
            lb = await client.get("/conversations", headers=hb)
            assert all(c["id"] != conv["id"] for c in lb.json())


async def test_http_requires_valid_token():
    await _seed_or_skip()
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            assert (await client.get("/conversations")).status_code == 401  # 无 token
            bad = {"Authorization": "Bearer not.a.jwt"}
            assert (await client.get("/conversations", headers=bad)).status_code == 401


async def test_http_cannot_login_user_across_tenant():
    """alice 的账号配 globex 的租户 → 登录失败（RLS 下查不到该租户的这个用户）。"""
    await _seed_or_skip()
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.post(
                "/auth/login",
                json={"tenant_slug": "globex", "email": "alice@acme.com", "password": "alice-pass"},
            )
            assert r.status_code == 401


# ============================ DB 层 RLS 直测 ============================


async def test_rls_enforced_at_db_layer():
    """用应用角色 trip_app 直连，证明 RLS 的 fail-closed / 隔离 / WITH CHECK。"""
    await _seed_or_skip()
    engine = create_async_engine(settings.app_database_url)  # trip_app（受 RLS）
    sm = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with sm() as s:
            ids = dict((await s.execute(select(Tenant.slug, Tenant.id))).all())
            await s.rollback()
        acme_id, globex_id = ids["acme"], ids["globex"]

        # fail-closed：没设租户上下文 → 一行都看不到
        async with sm() as s:
            assert len((await s.execute(select(User))).scalars().all()) == 0
            await s.rollback()

        # acme 上下文只看得到 acme 的用户
        async with sm() as s:
            await set_tenant_context(s, acme_id)
            emails = (await s.execute(select(User.email))).scalars().all()
            assert emails == ["alice@acme.com"]
            await s.rollback()

        # globex 上下文看不到 acme 的用户
        async with sm() as s:
            await set_tenant_context(s, globex_id)
            emails = (await s.execute(select(User.email))).scalars().all()
            assert "alice@acme.com" not in emails
            await s.rollback()

        # WITH CHECK：在 globex 上下文里写 acme 的行 → 被数据库拒绝
        async with sm() as s:
            await set_tenant_context(s, globex_id)
            s.add(Conversation(tenant_id=acme_id, user_id=acme_id, title="越权"))
            with pytest.raises(DBAPIError):
                await s.flush()
            await s.rollback()
    finally:
        await engine.dispose()
