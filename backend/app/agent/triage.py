"""分诊（triage）：意图分类 + 槽位抽取 + 澄清门控（M2+）。

为什么需要它（面试要点）：
  用户只说"我想去上海"时，必填信息（出发地/日期）根本不全。此时**正确行为不是
  瞎猜着调工具，而是反问一句**。M1/M2 我们靠 SYSTEM_PROMPT 的"信息不足先追问"
  让模型自觉做；这里把它升级成**显式、可控、可审计**的图节点：

    triage 用结构化输出抽出 (intent, 槽位) → 纯函数判断"槽位齐不齐" →
      齐 → 正常走 agent；缺 → 走 clarify 反问一句、结束本轮。

  好处：路由逻辑是确定性纯函数（可离线单测）；槽位状态可观测（能统计用户最常
  漏哪个槽）；triage 可用更便宜的小模型；高危流程能强制"槽位没齐不准下单"。
  代价：每轮多一次 LLM 调用——所以做成可开关（settings.enable_triage）。

这里只放**schema + 纯逻辑 + 提示词**；真正调 LLM 的 triage_node 是 graph.py 里
的闭包（要捕获 llm），就像 agent_node/tools_node 一样。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# 每种意图需要哪些"必填槽位"才能去调工具。空列表 = 不需要槽位即可放行
# （policy/expense 用户通常把要素写在问题里；chitchat 是闲聊）。
SLOT_REQUIREMENTS: dict[str, list[str]] = {
    "flight": ["origin", "destination", "date"],
    "train": ["origin", "destination", "date"],
    "hotel": ["destination", "date"],  # 入住城市 + 日期
    "weather": ["destination", "date"],
    "policy": [],
    "expense": [],
    "chitchat": [],
    "unknown": [],
}

# 缺某个"待澄清点"时，反问里对应的话术片段。
# purpose 是个特殊点：用户提了地点却没说想干嘛（"我想去上海"），先问清诉求。
SLOT_QUESTIONS: dict[str, str] = {
    "purpose": "你是想订机票、高铁，还是要查酒店/天气",
    "origin": "你从哪个城市出发",
    "destination": "你要去哪个城市",
    "date": "大概哪一天（或哪几天）",
}

TRIAGE_SYSTEM_PROMPT = """你是差旅助手的"意图与槽位抽取器"。请基于**整段对话**（不只是最后一句）判断：
1. intent：用户这次最核心的诉求属于哪一类（flight 机票 / train 高铁 / hotel 酒店 /
   weather 天气 / policy 报销政策 / expense 费用试算 / chitchat 闲聊 / unknown 不确定）。
2. 槽位：用户在对话里**明确提到**的出发城市 origin、目的城市 destination、出行日期 date。
   - 没提到的槽位一律留空（null），**绝对不要臆测或编造**。
   - date 只要用户给了任何时间（"明天""下周一""6月25"）就算提供了，原样填即可。
今天是 {today}。只做抽取，不要回答用户问题。"""


class TripIntent(BaseModel):
    """triage 的结构化输出 schema。用 with_structured_output 强制模型按它产出。"""

    intent: Literal[
        "flight", "train", "hotel", "weather", "policy", "expense", "chitchat", "unknown"
    ] = Field(description="用户本次最核心的意图分类")
    origin: str | None = Field(default=None, description="出发城市，未提到则 null")
    destination: str | None = Field(default=None, description="目的城市，未提到则 null")
    date: str | None = Field(default=None, description="出行日期或时间描述，未提到则 null")


def compute_missing_slots(intent: str, slots: dict[str, str | None]) -> list[str]:
    """纯函数：给定具体意图与已抽到的槽位，算出还缺哪些必填槽。
    槽位值为 None 或空串都算"没填"。"""
    required = SLOT_REQUIREMENTS.get(intent, [])
    return [s for s in required if not slots.get(s)]


def clarification_needs(intent: str, slots: dict[str, str | None]) -> list[str]:
    """纯函数：返回"需要向用户追问的点"列表；空列表 = 信息够了，放行去 agent。

    两种需要澄清的情形：
      1. 具体差旅意图(flight/train/hotel/weather) 但必填槽没齐 → 追问缺失槽；
      2. intent=unknown 却提到了出发/目的城市 → 这是个含糊的差旅请求（"我想去上海"
         没说坐什么、也没说哪天），先追问诉求(purpose) + 缺失的槽。
    其余情形（policy/expense/chitchat，或纯 unknown 没任何线索）直接放行。
    """
    if SLOT_REQUIREMENTS.get(intent):  # 具体意图且有必填槽
        return compute_missing_slots(intent, slots)
    if intent == "unknown" and (slots.get("destination") or slots.get("origin")):
        needs = ["purpose"]
        needs += [s for s in ("origin", "destination", "date") if not slots.get(s)]
        return needs
    return []


def build_clarify_question(needs: list[str]) -> str:
    """纯函数：根据待澄清点拼一句自然的反问。可离线单测。"""
    parts = [SLOT_QUESTIONS[s] for s in needs if s in SLOT_QUESTIONS]
    if not parts:
        return "能再补充一下你的差旅需求吗？比如出发地、目的地和日期。"
    return "为了帮你查得更准，先确认几点：" + "；".join(parts) + "？"


def route_after_triage(state) -> str:  # noqa: ANN001 —— LangGraph 条件边签名
    """条件边：triage 之后，有待澄清点就去 clarify 反问，否则正常走 agent。"""
    return "clarify" if state.get("clarify_needs") else "agent"
