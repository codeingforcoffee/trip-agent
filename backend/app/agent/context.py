"""上下文压缩（M6a）：把超预算的对话历史摘要成"滚动摘要"，限住 token 又不丢线索。

为什么需要：每轮都把全量历史重发，token 随轮数二次方增长（贵 + 慢），且历史越长模型越
"lost in the middle"（信噪比下降，反而答得差）。所以上下文是**预算**，要主动管理。

策略（M6+：**按窗口比例的双水位压缩**）：用量超过 high 水位（窗口的 70%）才触发；触发后从最老的
**完整轮次**淘汰，直到降到 low 水位（40%）以下，把淘汰的消息用一次便宜 LLM 调用**摘要**进 summary，
再用 LangGraph 的 RemoveMessage 移出历史。始终保留 system（agent 节点拼）+ 滚动摘要 + 最近若干轮原文。

为什么双水位而非"到阈值就压一点"：high/low 间的大 gap 让压缩后能撑很久才再触发，避免"压一点马上又超"
的抖动，也减少反复摘要的信息漂移。阈值按**窗口比例**而非绝对数，换模型自适应。

四个工程要点（面试会追）：
  1. **用真实用量当"当前占用"**：优先读上一轮 AIMessage.usage_metadata.total_tokens（天然含 system/
     摘要/记忆前缀），拿不到才回退字符估算——都只为触发判断，不求精确。
  2. **绝不拆散 tool_call 与它的 ToolMessage**：孤立的 tool_call 没有结果，下次喂 API 直接报错。
     所以裁剪以"完整轮次"为单位——保留窗口必须从一条 HumanMessage 开始，且有硬保底不压当前轮。
  3. **压缩是有损的**：low 水位只决定"压多少"，不决定"丢不丢"。"不丢关键上下文"靠的是【近轮保原文】+
     【关键事实提前结构化卸载】（triage 槽位、M6b 偏好在被摘要前就落库了），不是靠比例。
  4. **摘要要保住槽位**：明确要求摘要器留住出发地/目的地/日期/偏好等关键信息，
     这样压缩后 triage 仍能跨轮补槽、agent 仍知道用户要干什么。

本模块把"纯逻辑"（估算 + 选切点 + 产出移除指令）与"LLM 摘要"解耦：maybe_compress 接收一个
summarize 回调，于是可离线单测（喂假摘要器）；图里再用真 LLM 包一层。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

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


def latest_usage_tokens(messages: list[AnyMessage]) -> int | None:
    """从最近一条带用量元数据的 AIMessage 读**真实** token 数（上一轮实际喂进模型的量）。

    这是"当前窗口占用"最准的信号：它天然把 system/摘要/记忆等注入前缀都算进去了
    （都是那次真实 API 调用的一部分），比字符估算靠谱。拿不到（离线/测试/模型未回传）→ None，
    由调用方回退到 estimate_tokens。
    """
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            usage = getattr(m, "usage_metadata", None)
            if usage and usage.get("total_tokens"):
                return int(usage["total_tokens"])
    return None


def _watermark_cut(messages: list[AnyMessage], low_tokens: int, keep_last_floor: int) -> int:
    """选切点 cut：从最老的**完整轮次**淘汰，直到 messages[cut:] 的 token ≤ low_tokens。

    约束：① cut 必落在 HumanMessage 边界（保留窗口从一轮开头起，绝不拆散 tool_call 与其 ToolMessage）；
    ② 至少保留 keep_last_floor 条原文（硬保底，绝不淘汰当前轮）。
    策略：**淘汰最少**就能回到 low 的那个切点（信息损失最小）；若连保底之外全淘汰也到不了 low
    （近段是超大工具结果），就尽力淘汰到保底允许的最老边界（可能仍高于 low，best effort）。
    """
    max_cut = max(0, len(messages) - keep_last_floor)
    humans = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage) and i <= max_cut]
    best = 0
    for cut in humans:  # 升序：cut 越大 → 淘汰越多、保留越少
        best = cut
        if estimate_tokens(messages[cut:]) <= low_tokens:
            return cut  # 淘汰最少即达标，停
    return best  # 达不到 low，淘汰到保底允许的最老边界


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
    window_tokens: int,
    high_ratio: float,
    low_ratio: float,
    keep_last_floor: int,
    summarize: Summarizer,
    used_tokens: int | None = None,
) -> CompressResult | None:
    """双水位压缩：用量超过 high 水位才压，压到 low 水位以下。否则返回 None（充足→不动）。

    used_tokens：当前窗口占用；调用方可传真实用量（latest_usage_tokens），None 时回退到字符估算。
    纯编排逻辑，不绑定具体 LLM：summarize 回调负责真正生成摘要（图里传真 LLM，测试里传假的）。
    """
    used = used_tokens if used_tokens is not None else estimate_tokens(messages)
    if used <= window_tokens * high_ratio:
        return None  # 未过 high 水位：上下文充足，不压
    cut = _watermark_cut(messages, int(window_tokens * low_ratio), keep_last_floor)
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
