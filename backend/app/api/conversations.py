"""会话增查（M3）：用来【演示多租户隔离】的最小载体。

关键观察：这两个接口的代码里**没有一句 `WHERE tenant_id = ...`**——
租户过滤完全由 RLS 在数据库层兜底。create 时 RLS 的 WITH CHECK 还会拒绝
写入不属于当前租户上下文的行。这正是"应用层不写过滤、也不会泄露"的纵深防御演示。

M9 会在此基础上加 /chat（SSE 流式），并把 identity → config 注入 LangGraph。
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_identity, get_tenant_session
from app.core.security import Identity
from app.db.models import Conversation

router = APIRouter(prefix="/conversations", tags=["conversations"])


class CreateConversation(BaseModel):
    title: str = "新会话"


class ConversationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    title: str
    user_id: UUID
    tenant_id: UUID
    created_at: datetime


@router.post("", response_model=ConversationOut)
async def create_conversation(
    body: CreateConversation,
    identity: Identity = Depends(get_identity),
    session: AsyncSession = Depends(get_tenant_session),
) -> Conversation:
    # tenant_id/user_id 来自【可信的 Identity】，不来自请求体——客户端无法伪造归属
    conv = Conversation(
        tenant_id=identity.tenant_id,
        user_id=identity.user_id,
        title=body.title,
    )
    session.add(conv)
    await session.flush()  # 触发 INSERT，拿回 server_default 生成的 id
    await session.refresh(conv)  # 读回 created_at 等数据库侧默认值
    return conv


@router.get("", response_model=list[ConversationOut])
async def list_conversations(
    session: AsyncSession = Depends(get_tenant_session),
) -> list[Conversation]:
    # 注意：没有任何租户过滤条件，可见性全靠 RLS。换个租户的 token 来查，结果集自动不同。
    rows = (
        (await session.execute(select(Conversation).order_by(Conversation.created_at.desc())))
        .scalars()
        .all()
    )
    return list(rows)
