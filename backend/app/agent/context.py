"""上下文压缩（M6a）：把超预算的对话历史摘要成"滚动摘要"，限住 token 又不丢线索。

为什么需要：每轮都把全量历史重发，token 随轮数二次方增长（贵 + 慢），且历史越长模型越
"lost in the middle"（信噪比下降，反而答得差）。所以上下文是**预算**，要主动管理。

策略：超阈值时，保留 system（在 agent 节点拼）+ 滚动摘要 + 最近 N 条原文；把更早的消息
用一次便宜 LLM 调用**摘要**进 summary，再用 LangGraph 的 RemoveMessage 真正移出历史。

三个工程要点（面试会追）：
  1. **token 不必精确**：触发判断用估算即可（真实用量可读 AIMessage.response_metadata.token_usage）。
  2. **绝不拆散 tool_call 与它的 ToolMessage**：孤立的 tool_call 没有结果，下次喂 API 直接报错。
     所以裁剪以"完整轮次"为单位——保留窗口必须从一条 HumanMessage 开始。
  3. **摘要要保住槽位**：明确要求摘要器留住出发地/目的地/日期/偏好等关键信息，
     这样压缩后 triage 仍能跨轮补槽、agent 仍知道用户要干什么。

本模块把"纯逻辑"（估算 + 选切点 + 产出移除指令）与"LLM 摘要"解耦：maybe_compress 接收一个
summarize 回调，于是可离线单测（喂假摘要器）；图里再用真 LLM 包一层。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from langchain_core.messages import AnyMessage, HumanMessage

# 摘要器签名：给定（待并入的历史消息, 旧摘要）→ 返回新的滚动摘要文本
Summarizer = Callable[[list[AnyMessage], str], Awaitable[str]]


def _estimate_text_tokens(text: str) -> int:
    """粗估一段文本的 token 数。中文约 0.6 token/字，其他约 0.25 token/字符——只为触发判断，不求精确。"""
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
    other = len(text) - cjk
    return int(cjk * 0.6 + other * 0.25)


def _message_text(m: AnyMessage) -> str:
    """取一条消息用于估算/摘要的文本：正文 + 工具调用的参数（这些也占 token）。"""
    text = str(m.content or "")
    for call in getattr(m, "tool_calls", None) or []:
        text += " " + str(call.get("name", "")) + str(call.get("args", ""))
    return text


def estimate_tokens(messages: list[AnyMessage]) -> int:
    """估算一组消息的总 token（含每条 ~4 token 的角色/结构开销）。"""
    return sum(_estimate_text_tokens(_message_text(m)) + 4 for m in messages)


def _safe_cut(messages: list[AnyMessage], keep_last: int) -> int:
    """选一个**安全切点** cut：messages[:cut] 淘汰、messages[cut:] 保留。

    要求：保留窗口必须从一条 HumanMessage 开始——这样既不会把某轮的 tool_call 与其
    ToolMessage 拆散，也保证保留的是若干**完整轮次**。做法：从"末尾留 keep_last 条"的目标位
    向前找最近的一条 HumanMessage 作为切点；找不到（如近段全在一个长轮次里）则返回 0 不压缩。
    """
    target = max(0, len(messages) - keep_last)
    for i in range(target, -1, -1):
        if i < len(messages) and isinstance(messages[i], HumanMessage):
            return i
    return 0


@dataclass(frozen=True)
class CompressResult:
    """压缩产出：新滚动摘要 + 待移除消息的 id 列表 + 被摘要掉的消息数（用于日志/断言）。"""

    summary: str
    removed_ids: list[str]
    evicted_count: int


async def maybe_compress(
    messages: list[AnyMessage],
    summary: str,
    *,
    budget: int,
    keep_last: int,
    summarize: Summarizer,
) -> CompressResult | None:
    """若历史超预算则压缩，返回 CompressResult；否则返回 None（不动）。

    纯编排逻辑，不绑定具体 LLM：summarize 回调负责真正生成摘要（图里传真 LLM，测试里传假的）。
    """
    if estimate_tokens(messages) <= budget:
        return None  # 没超预算，pass-through
    cut = _safe_cut(messages, keep_last)
    evicted = messages[:cut]
    # 只移除带 id 的消息（add_messages 入库的消息都有 id）；没有可淘汰的就不压缩
    removable = [m for m in evicted if getattr(m, "id", None)]
    if not removable:
        return None
    new_summary = await summarize(evicted, summary)
    return CompressResult(
        summary=new_summary,
        removed_ids=[m.id for m in removable],
        evicted_count=len(evicted),
    )


# —— 真实 LLM 摘要器 ——

_SUMMARY_PROMPT = """你是对话压缩器。把【旧摘要】与【待并入的历史消息】融合成一份更新后的滚动摘要。
要求：
- 第三人称、简洁地记录：用户的核心意图；已明确的关键信息（出发地/目的地/日期/人数/预算/偏好等）；
  已调用的工具及其结论；尚未解决的问题。
- 不要编造未出现的信息，不保留寒暄客套，控制在 200 字以内。

【旧摘要】
{old}

【待并入的历史消息】
{convo}

直接输出更新后的摘要正文（不要前缀、不要解释）："""


def make_llm_summarizer(llm) -> Summarizer:
    """用聊天模型构造一个摘要器回调，供图里的 compress 节点使用。"""

    async def _summarize(evicted: list[AnyMessage], old_summary: str) -> str:
        convo = "\n".join(f"{type(m).__name__}: {_message_text(m)}" for m in evicted)
        prompt = _SUMMARY_PROMPT.format(old=old_summary or "（无）", convo=convo)
        resp = await llm.ainvoke([HumanMessage(content=prompt)])
        return str(resp.content).strip()

    return _summarize
