"""灌入演示数据（M3）：2 个租户 + 各自用户。

用【超级用户 trip】连库（settings.database_url）——超级用户无视 RLS，所以能跨租户
自由写入。这正是"管理操作用超级用户、应用运行时用普通角色 trip_app"分工的体现。

运行：cd backend && uv run python scripts/seed.py
幂等：按 slug / (tenant,email) 找不到才插入，可反复跑。
"""

from __future__ import annotations

import asyncio

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import settings
from app.core.security import SCOPE_BOOKING, SCOPE_CHAT, hash_password
from app.db.models import Tenant, User

# 演示用账号：两个互不相干的企业租户。密码明文仅用于演示登录，入库存的是 bcrypt 哈希。
_SEED = [
    {
        "tenant": {"name": "Acme 差旅", "slug": "acme"},
        "users": [
            {
                "email": "alice@acme.com",
                "password": "alice-pass",
                "display_name": "Alice（Acme 员工）",
                "scopes": [SCOPE_CHAT, SCOPE_BOOKING],  # 能对话 + 能下单
            }
        ],
    },
    {
        "tenant": {"name": "Globex 出行", "slug": "globex"},
        "users": [
            {
                "email": "bob@globex.com",
                "password": "bob-pass",
                "display_name": "Bob（Globex 员工）",
                "scopes": [SCOPE_CHAT],  # 只能对话，不能下单（M7 演示越权拦截）
            }
        ],
    },
]


async def _get_or_create_tenant(session, name: str, slug: str) -> Tenant:
    tenant = (await session.execute(select(Tenant).where(Tenant.slug == slug))).scalar_one_or_none()
    if tenant is None:
        tenant = Tenant(name=name, slug=slug)
        session.add(tenant)
        await session.flush()
    return tenant


async def _get_or_create_user(session, tenant: Tenant, spec: dict) -> tuple[User, bool]:
    user = (
        await session.execute(
            select(User).where(User.tenant_id == tenant.id, User.email == spec["email"])
        )
    ).scalar_one_or_none()
    if user is not None:
        return user, False
    user = User(
        tenant_id=tenant.id,
        email=spec["email"],
        password_hash=hash_password(spec["password"]),
        display_name=spec["display_name"],
        scopes=spec["scopes"],
    )
    session.add(user)
    await session.flush()
    return user, True


async def main() -> None:
    # 超级用户连接（无视 RLS）——管理操作专用
    engine = create_async_engine(settings.database_url)
    sm = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with sm() as session:
            for entry in _SEED:
                tenant = await _get_or_create_tenant(session, **entry["tenant"])
                for uspec in entry["users"]:
                    user, created = await _get_or_create_user(session, tenant, uspec)
                    flag = "＋新建" if created else "已存在"
                    print(
                        f"[{flag}] 租户 {tenant.slug:7s} ({tenant.id}) "
                        f"用户 {user.email} scopes={uspec['scopes']} 密码={uspec['password']}"
                    )
            await session.commit()
        print("\n种子数据就绪。登录示例：")
        print(
            '  POST /auth/login {"tenant_slug":"acme",  "email":"alice@acme.com",  "password":"alice-pass"}'
        )
        print(
            '  POST /auth/login {"tenant_slug":"globex","email":"bob@globex.com",  "password":"bob-pass"}'
        )
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
