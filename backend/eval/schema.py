"""评测数据结构：黄金场景（输入）+ 运行轨迹 Trace + 单场景结果（输出）。

Scenario 从 datasets/*.jsonl 逐行解析（pydantic 做校验 + 填默认，缺字段不炸）。
所有"期望"集中在 expect 子对象里，按需填——查询类只填 tools_include/max_steps，
安全类填 audit_include/not_executed/final_excludes，RAG 类填 rag_should_cite/final_contains。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field


class Expect(BaseModel):
    """一条场景的**确定性**期望（能断言的都放这，不劳烦 LLM judge）。"""

    tools_include: list[str] = Field(default_factory=list)  # 这些工具必须被调用（recall）
    tools_exclude: list[str] = Field(
        default_factory=list
    )  # 这些工具绝不能被调用（如查询里混进下单）
    max_steps: int | None = None  # 推理步数上界（效率）：agent 决策轮数不得超过它
    audit_include: list[str] = Field(default_factory=list)  # 必须出现的 audit 动作
    not_executed: list[str] = Field(
        default_factory=list
    )  # 绝不能真正执行的高危工具（查 tool.executed）
    final_contains: list[str] = Field(default_factory=list)  # 最终答复必须包含的子串
    final_excludes: list[str] = Field(
        default_factory=list
    )  # 最终答复绝不能包含（如原始手机号=PII 泄漏）
    memory_recall_contains: list[str] = Field(default_factory=list)  # 召回的记忆上下文必须包含
    rag_should_cite: bool = False  # 是否要求答案标注政策来源（judge 兜底看忠实度）


class Scenario(BaseModel):
    id: str
    category: str  # booking | rag | safety | tools | memory
    input: str
    principal: str = "alice"  # alice(acme,有 booking:write) | bob(globex,仅 chat:write)
    # —— 图开关（按场景需要开）——
    triage: bool = False  # 默认关：triage 会多一次 LLM 调用、也影响步数，只有测澄清才开
    memory: bool = False  # 记忆召回场景才开
    hitl: bool = True  # 默认开 HITL；下单类会中断，用 hitl_approve 自动决策
    hitl_approve: bool = True
    seed_memory: str | None = None  # 记忆场景：跑前先 seed 这条语义记忆
    rubric: str = ""  # 交给 judge 的评分说明；空则跳过质量评分
    expect: Expect = Field(default_factory=Expect)


@dataclass
class Trace:
    """一次场景运行的**可观测轨迹**——确定性指标全部从这里算，不再碰 LLM。"""

    tools_called: list[str] = field(default_factory=list)  # 按调用顺序（可含重复）
    executed_high_risk: list[str] = field(
        default_factory=list
    )  # 真正执行的高危工具（audit tool.executed）
    steps: int = 0  # agent 决策轮数（ReAct 迭代次数）
    audits: list[dict] = field(default_factory=list)  # [{action, detail}]
    tool_outputs: list[str] = field(
        default_factory=list
    )  # 工具返回文本（供 judge 判 faithfulness）
    final: str = ""  # 最终（脱敏后）答复文本
    memory_context: str = ""  # 本轮召回注入的记忆上下文
    tokens: int = 0  # 累计 token（record/live 为真实用量；replay 用录下的）
    interrupted: bool = False  # 是否触发过 HITL 中断
    error: str | None = None  # 运行异常（判为 fail 而非崩溃整场评测）

    @property
    def audit_actions(self) -> set[str]:
        return {a["action"] for a in self.audits}


@dataclass
class Check:
    """一条确定性检查的结果。applicable=False 表示该场景不涉及此项（不计入通过率）。"""

    name: str
    applicable: bool
    passed: bool
    detail: str = ""


@dataclass
class ScenarioResult:
    scenario_id: str
    category: str
    checks: list[Check] = field(default_factory=list)
    steps: int = 0
    tokens: int = 0
    tool_recall: float | None = None  # include 覆盖率
    tool_precision_ok: bool | None = None  # 没有调用 exclude 里的工具
    judge_quality: int | None = None  # 1~5
    judge_faithful: bool | None = None
    judge_reason: str = ""
    error: str | None = None

    @property
    def applicable_checks(self) -> list[Check]:
        return [c for c in self.checks if c.applicable]

    @property
    def passed(self) -> bool:
        """场景通过 = 所有**适用的**确定性检查都过（judge 质量单列，不卡通过）。"""
        if self.error:
            return False
        applicable = self.applicable_checks
        return bool(applicable) and all(c.passed for c in applicable)
