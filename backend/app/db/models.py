"""多租户 ORM 模型（M3）。

设计主线：**每一张业务表都带 tenant_id，租户是数据的第一维度**。
这是「共享库共享表（pooled）」隔离模型——成本最低、利用率最高，代价是
"漏一个过滤条件就跨租户泄露"。所以我们不只靠应用层过滤，还在数据库层加
行级安全（RLS，见 Alembic 迁移），形成纵深防御。

几个刻意的选择（面试可讲）：
  - **UUID 主键**而非自增整数：① 跨租户不可枚举（自增 id 会暴露"系统里有多少条"，
    且能被猜测遍历）；② 分布式友好（无需中心发号）。代价是索引比 bigint 略大。
  - **命名约定（naming_convention）**：给约束/索引统一命名规则，让 Alembic 自动迁移
    生成的名字稳定可预测，避免不同机器生成出不同名字导致迁移漂移。
  - **scopes 存数组**：用户的权限范围（如 booking:write），M7 工具授权会读它。
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    ForeignKey,
    MetaData,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# 统一的约束命名规则：%(table_name)s / %(column_0_name)s 等是 SQLAlchemy 的占位符。
# 没有它，Alembic autogenerate 出来的约束名可能因环境而异，迁移历史不可复现。
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """所有模型的基类。统一在这里挂命名约定。"""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


def _uuid_pk() -> Mapped[uuid.UUID]:
    """UUID 主键列工厂：服务端用 gen_random_uuid() 生成（PG13+ 内置）。

    server_default 让数据库负责生成默认值——即使有人不经 ORM 直接 INSERT，
    主键也不会缺失（单一事实来源在 DB，而非应用代码）。
    """
    return mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )


def _created_at() -> Mapped[datetime]:
    return mapped_column(server_default=func.now(), nullable=False)


class Tenant(Base):
    """租户（一个企业客户）。**它本身不带 tenant_id**——它就是租户维度的根。"""

    __tablename__ = "tenants"

    id: Mapped[uuid.UUID] = _uuid_pk()
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    slug: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    # 隔离档位：pooled（默认共享）/ silo（独立库，给受监管大客户）。
    # 现在全是 pooled；留这个字段是把"隔离是分档商业决策"显式建模出来，
    # 将来要把某租户迁到独立库，先改这里再路由到不同连接。
    isolation_tier: Mapped[str] = mapped_column(String(20), nullable=False, default="pooled")
    created_at: Mapped[datetime] = _created_at()

    users: Mapped[list[User]] = relationship(back_populates="tenant")


class User(Base):
    """用户（隶属某租户的员工）。身份的载体：登录后签进 JWT 的就是 (tenant_id, id, scopes)。"""

    __tablename__ = "users"
    # 同一租户内 email 唯一；不同租户可以有同名 email（租户是命名空间）——
    # 所以唯一约束是 (tenant_id, email) 复合，而非 email 单列。
    __table_args__ = (UniqueConstraint("tenant_id", "email", name="uq_users_tenant_email"),)

    id: Mapped[uuid.UUID] = _uuid_pk()
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    email: Mapped[str] = mapped_column(String(255), nullable=False)
    # bcrypt 哈希后的密码（绝不存明文）。存为字符串，含算法/工作因子/盐，自描述。
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    display_name: Mapped[str] = mapped_column(String(200), nullable=False, default="")
    # 权限范围（如 ["chat:write", "booking:write"]），M7 工具授权据此放行/拒绝。
    scopes: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, default=list)
    created_at: Mapped[datetime] = _created_at()

    tenant: Mapped[Tenant] = relationship(back_populates="users")


class Conversation(Base):
    """会话（一个对话线程）。thread_id = {tenant_id}:{user_id}:{conv_id} 的 conv_id 就是它。"""

    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = _uuid_pk()
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    title: Mapped[str] = mapped_column(String(500), nullable=False, default="新会话")
    created_at: Mapped[datetime] = _created_at()


class UserPreference(Base):
    """用户长期偏好（M6b）：跨会话记住的**结构化**偏好，按 (tenant,user,key) 唯一。

    为什么偏好用结构化表而非向量库（面试要点）：偏好会**变**（"靠窗"→"靠过道"），
    结构化 + 唯一键让"更新"就是一次幂等 upsert（recency wins），绝不产生自相矛盾的两条；
    若塞进向量库，新旧值会并存，召回可能给出过时的那条。自由文本的情景事实才进 Qdrant。

    confidence/source 让"用户明说的"（高置信）与"系统推断的"（低置信）可区分——
    避免一次行为就过拟合成规则。updated_at 支持"近时优先"。
    """

    __tablename__ = "user_preferences"
    __table_args__ = (
        UniqueConstraint("tenant_id", "user_id", "key", name="uq_user_preferences_tenant_user_key"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    key: Mapped[str] = mapped_column(String(100), nullable=False)  # 归一化键，如 seat_preference
    value: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(nullable=False, default=1.0)
    source: Mapped[str] = mapped_column(
        String(20), nullable=False, default="explicit"
    )  # explicit/inferred
    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(), onupdate=func.now(), nullable=False
    )


class AuditLog(Base):
    """审计日志（M7 大用，M3 先把表建好）。谁、在哪个租户、做了什么、细节。

    安全合规的基础：高危动作（下单/取消/导出）、工具调用、越权尝试都往这里写，
    可追溯、不可抵赖。detail 用 JSONB 存结构化上下文，便于按字段查询。
    """

    __tablename__ = "audit_logs"

    id: Mapped[uuid.UUID] = _uuid_pk()
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    detail: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    created_at: Mapped[datetime] = _created_at()


# 受 RLS 保护的表清单（带 tenant_id 的业务表）。测试引用它，避免"新增一张表忘了开 RLS"。
# 注意：历史迁移各自【冻结】自己那批表名（不引用本常量），以保证从零重放时顺序正确
# （M3 建前三张、M6b 建 user_preferences，各自开各自的 RLS）。
TENANT_SCOPED_TABLES = ("users", "conversations", "audit_logs", "user_preferences")
