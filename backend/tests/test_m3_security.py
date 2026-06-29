"""M3 离线单测：密码哈希 / JWT / Identity / config 注入。

全部不依赖数据库与网络——锁住"可信身份"这层的纯逻辑。
"""

from __future__ import annotations

from uuid import uuid4

import jwt
import pytest

from app.agent.identity import build_runnable_config
from app.core.security import (
    SCOPE_BOOKING,
    SCOPE_CHAT,
    Identity,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


# ---------- 密码哈希 ----------


def test_password_hash_roundtrip():
    h = hash_password("s3cret")
    assert h != "s3cret"  # 绝不明文
    assert h.startswith("$2")  # bcrypt 哈希前缀（自描述算法/成本/盐）
    assert verify_password("s3cret", h) is True
    assert verify_password("wrong", h) is False


def test_password_same_input_different_hash():
    # 盐随机 → 同明文两次哈希不同（但都能校验通过）。这是防彩虹表的基础。
    assert hash_password("x") != hash_password("x")


def test_verify_handles_garbage_hash():
    assert verify_password("x", "not-a-bcrypt-hash") is False  # 不抛，返回 False


# ---------- JWT ----------


def _identity() -> Identity:
    return Identity(
        tenant_id=uuid4(),
        user_id=uuid4(),
        email="a@b.com",
        scopes=(SCOPE_CHAT, SCOPE_BOOKING),
    )


def test_jwt_roundtrip_preserves_identity():
    ident = _identity()
    token = create_access_token(ident)
    back = decode_access_token(token)
    assert back.tenant_id == ident.tenant_id
    assert back.user_id == ident.user_id
    assert back.email == ident.email
    assert back.scopes == ident.scopes  # tuple 顺序一致


def test_jwt_tampered_signature_rejected():
    token = create_access_token(_identity())
    with pytest.raises(jwt.PyJWTError):
        decode_access_token(token + "tamper")


def test_jwt_expired_rejected():
    token = create_access_token(_identity(), expires_minutes=-1)  # 已过期
    with pytest.raises(jwt.ExpiredSignatureError):
        decode_access_token(token)


# ---------- Identity / config ----------


def test_identity_thread_id_namespacing():
    tid, uid, cid = uuid4(), uuid4(), uuid4()
    ident = Identity(tenant_id=tid, user_id=uid)
    assert ident.thread_id(cid) == f"{tid}:{uid}:{cid}"  # 租户在最前


def test_identity_has_scope():
    ident = Identity(tenant_id=uuid4(), user_id=uuid4(), scopes=(SCOPE_CHAT,))
    assert ident.has_scope(SCOPE_CHAT)
    assert not ident.has_scope(SCOPE_BOOKING)


def test_build_runnable_config_shape():
    ident = _identity()
    conv_id = uuid4()
    cfg = build_runnable_config(ident, conv_id)
    conf = cfg["configurable"]
    assert conf["thread_id"] == f"{ident.tenant_id}:{ident.user_id}:{conv_id}"
    assert conf["tenant_id"] == str(ident.tenant_id)
    assert conf["user_id"] == str(ident.user_id)
    assert conf["scopes"] == list(ident.scopes)
