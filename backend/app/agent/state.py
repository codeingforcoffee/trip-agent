"""Agent 的共享状态（State）。

LangGraph 的图就是"一个 State 在节点间流动、被不断更新"。每个节点接收当前
State、返回一个"部分更新"，框架按字段的 reducer 合并进 State。

M1 只需要一个字段 messages（对话消息列表）：
  - 类型注解里的 `Annotated[..., add_messages]` 指定了它的 reducer：
    节点返回的新消息会被**追加**到现有列表，而不是整列表覆盖。
    add_messages 还会按 message.id 去重/更新，并自动处理 ToolMessage 配对。
  - 这也是为什么节点只需 `return {"messages": [新消息]}`，框架替你 append。

后续里程碑会往这里加字段（都各带自己的 reducer），例如：
  - summary: str          # M6 上下文压缩后的滚动摘要
  - tenant_id/user_id     # M3 多租户身份（也可放 config.configurable，见 graph.py 说明）
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # 对话消息列表；reducer=add_messages 表示"追加合并"
    messages: Annotated[list[AnyMessage], add_messages]
