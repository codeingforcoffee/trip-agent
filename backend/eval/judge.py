"""LLM-as-judge：只评**确定性断言评不了**的开放式质量——答得对不对、忠不忠于证据。

判官偏差控制（面试要点）：
  - 结构化输出（强制 schema）+ rubric 写死评分锚点 → 减少自由发挥；
  - 温度 0（在 llm 侧设）→ 尽量可复现；judge 调用同样走 cassette，回放时完全确定。
  - faithfulness 专门喂**工具/检索返回的原文**，让判官核对"答案有没有超出证据瞎编"——
    这是 RAG 的命门（宁可说"没查到"，不可编金额）。

判官只产出参考分（quality 1~5 / faithful），**不决定场景通过与否**（通过由确定性检查裁定）——
避免把"非确定的判官"塞进回归的关键路径。
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from eval.schema import Scenario, Trace


class JudgeVerdict(BaseModel):
    quality: int = Field(ge=1, le=5, description="答案整体质量 1~5（对照 rubric）")
    faithful: bool = Field(description="答案是否忠于工具/检索返回的证据（无捏造）")
    reason: str = Field(description="一句话评语")


_JUDGE_PROMPT = """你是严格的评测判官。依据给定 rubric 与证据，对助手答案打分。只看事实与 rubric，不看措辞华丽。

【用户请求】
{input}

【工具/检索返回的证据（判 faithful 的依据；可能为空）】
{evidence}

【助手最终答复】
{answer}

【评分 rubric】
{rubric}

打分要求：
- quality：1~5，是否满足 rubric 的核心要求（满足关键点给高分，答非所问/缺关键信息给低分）。
- faithful：答案是否**不超出证据**。若证据为空而答案凭空给出具体金额/条款 → false；如实说"未找到"→ true。
- reason：一句话说明扣分点或亮点。"""


async def judge(scenario: Scenario, trace: Trace, judge_llm) -> JudgeVerdict | None:
    """对一个场景做质量评分；无 rubric 则跳过（返回 None）。judge_llm 由调用方注入（可回放）。"""
    if not scenario.rubric:
        return None
    evidence = "\n---\n".join(trace.tool_outputs) if trace.tool_outputs else "（无）"
    prompt = _JUDGE_PROMPT.format(
        input=scenario.input,
        evidence=evidence[:2000],
        answer=trace.final or "（空）",
        rubric=scenario.rubric,
    )
    structured = judge_llm.with_structured_output(JudgeVerdict)
    return await structured.ainvoke([HumanMessage(content=prompt)])
