"""身份 → LangGraph 运行时配置（M3）。

把请求边界解析出的 Identity 装进 LangGraph 的 `config.configurable`，而**不是**塞进
AgentState。原因（面试核心）：
  - State 是【对话内容】：会被 checkpointer 持久化、会喂进 LLM 上下文、随对话演化。
  - config.configurable 是【每次调用的运行时上下文】：不持久化进对话、不进 prompt、
    由可信的调用方设定。身份正属于后者。
  - 安全：身份若进 State/messages，prompt injection 理论上能诱导模型在工具参数里改
    tenant_id；放 config 则身份待在模型碰不到的可信层，工具/节点从 config 读取。
  - 机制：LangGraph 节点和工具都能拿到 config；checkpointer 也用 thread_id 做命名空间。

M9 的 /chat 路由会：identity = Depends(get_identity) → config = build_runnable_config(...)
→ graph.ainvoke(state, config)。M3 先把这个翻译层和约定立好，并单测。
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from app.core.security import Identity


def build_runnable_config(identity: Identity, conv_id: UUID | str) -> dict[str, Any]:
    """构造传给 graph.ainvoke 的 config。

    thread_id = {tenant}:{user}:{conv}（短期记忆按它命名空间隔离）；
    tenant_id/user_id/scopes 也一并放进去，供工具做租户过滤 / 权限校验时读取。
    """
    return {
        "configurable": {
            "thread_id": identity.thread_id(conv_id),
            "tenant_id": str(identity.tenant_id),
            "user_id": str(identity.user_id),
            "scopes": list(identity.scopes),
        }
    }
