"""身份与凭证（M3）：密码哈希 + JWT 签发/校验 + Identity 值对象。

三件事，对应面试三个考点：
  1. **密码哈希（bcrypt）**：绝不存明文；bcrypt 自带盐 + 工作因子（计算成本），
     哈希串自描述（含算法/成本/盐）；校验用恒定时间比较，抗时序侧信道。
  2. **JWT**：无状态身份令牌。服务端用密钥签名，客户端带着它来；服务端只验签名
     就能信任里面的 (tenant_id, user_id, scopes)，无需查库（可水平扩展的关键）。
  3. **Identity**：从 token 解出的不可变身份值对象。它会被注入 LangGraph 的
     config.configurable（不是 State）——身份是可信上下文，不是对话内容（防注入篡改）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from uuid import UUID

import bcrypt
import jwt

from app.core.config import settings

# —— 常用权限范围（scope）。M7 工具授权会校验它 ——
SCOPE_CHAT = "chat:write"  # 能对话
SCOPE_BOOKING = "booking:write"  # 能下单/取消（高危）


# ============================ 密码哈希 ============================


def hash_password(plain: str) -> str:
    """bcrypt 哈希。gensalt() 默认 cost=12（2^12 次迭代），随硬件升级可调高。"""
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """恒定时间校验明文与哈希是否匹配。哈希损坏/格式非法时返回 False，不抛。"""
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except ValueError:
        return False


# ============================ Identity ============================


@dataclass(frozen=True)
class Identity:
    """从 JWT 解出的不可变身份。贯穿请求 → DB（RLS 租户上下文）→ Agent（config）。"""

    tenant_id: UUID
    user_id: UUID
    email: str = ""
    scopes: tuple[str, ...] = field(default_factory=tuple)

    def has_scope(self, scope: str) -> bool:
        return scope in self.scopes

    def thread_id(self, conv_id: UUID | str) -> str:
        """LangGraph 线程命名空间：{tenant}:{user}:{conv}。

        租户放最前：同租户的会话天然聚簇（前缀扫描友好），也让"会话不串租户"
        在命名层面就成立——checkpoint 短期记忆据此隔离。
        """
        return f"{self.tenant_id}:{self.user_id}:{conv_id}"


# ============================ JWT ============================


def create_access_token(identity: Identity, expires_minutes: int | None = None) -> str:
    """签发短期访问令牌。载荷只放身份必需项，不放敏感数据（JWT 默认可被解码读取）。"""
    now = datetime.now(timezone.utc)
    exp = now + timedelta(minutes=expires_minutes or settings.jwt_expire_minutes)
    payload = {
        "sub": str(identity.user_id),  # subject = 用户
        "tenant": str(identity.tenant_id),
        "email": identity.email,
        "scopes": list(identity.scopes),
        "iat": now,  # 签发时间
        "exp": exp,  # 过期时间（pyjwt 会在 decode 时自动校验）
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> Identity:
    """验签 + 解码，还原 Identity。签名错/过期/格式非法都会抛 jwt.PyJWTError，
    由上层依赖捕获转成 401（见 core/deps.py）。"""
    payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    return Identity(
        tenant_id=UUID(payload["tenant"]),
        user_id=UUID(payload["sub"]),
        email=payload.get("email", ""),
        scopes=tuple(payload.get("scopes", [])),
    )
