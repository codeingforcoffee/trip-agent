"""差旅/报销政策查询工具（M5：关键词匹配 → 真 RAG 检索 + 引用）。

这是"工具是稳定接口、实现可演进"的样板：名字 `query_travel_policy`、对模型暴露的入参
（一个 question）从 M2 到 M5 完全没变，但实现从"内置关键词表"换成了"向量检索本租户政策文档"。
图、bind_tools、上层逻辑一行都不用改。

两个 M5 的关键设计：
  1. **租户从 config 注入，不从 LLM 参数取**：tenant_id 取自 LangGraph 的 config.configurable
     （M3 注入的可信身份），而非模型传进来的参数——所以模型即便被注入也无法越权查别家政策。
     机制：函数声明一个 `config: RunnableConfig` 形参，LangChain 自动注入且**不**把它暴露进
     给模型看的入参 schema。
  2. **带引用 + 检索不到就说没有**：把命中的条款连同来源/章节回灌给模型，并在 system prompt
     里要求据此作答、标注来源；检索为空时返回"未找到"，而不是硬塞内容诱导模型编造（faithfulness）。
"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.rag.retriever import RetrievedChunk, retrieve

log = get_logger("app.agent.tools.policy")


class PolicyQuery(BaseModel):
    question: str = Field(
        min_length=1, description="关于差旅或报销政策的问题，如「住宿费报销上限多少」"
    )


def _format_hits(hits: list[RetrievedChunk]) -> str:
    """把检索命中拼成带编号引用的文本，供模型据此回答并标注来源。"""
    lines = ["已根据公司差旅政策检索到以下条款，请严格据此回答并标注来源："]
    for i, h in enumerate(hits, 1):
        section = f" §{h.section}" if h.section else ""
        # chunk 正文带了"标题路径："前缀，展示时去掉首行前缀避免与引用标签重复
        body = h.text.split("\n", 1)[1] if "\n" in h.text else h.text
        lines.append(f"\n【来源{i}：{h.source}{section}】\n{body.strip()}")
    return "\n".join(lines)


@tool(args_schema=PolicyQuery)
async def query_travel_policy(question: str, config: RunnableConfig) -> str:
    """查询公司差旅与报销政策（住宿/机票/高铁/餐补/市内交通的标准与流程）。

    适用场景：用户问「能报多少」「需要什么发票」「什么职级能坐公务舱」等政策类问题。
    本工具基于公司政策文档做检索，返回带来源引用的相关条款；
    回答政策金额/规则时必须依据本工具的返回，不要凭空编造。
    """
    # config.configurable 由 M3 的 build_runnable_config 注入；缺租户身份则 fail-closed
    tenant_id = (config.get("configurable") or {}).get("tenant_id")
    if not tenant_id:
        log.warning("policy.no_tenant")
        return "无法确定当前租户身份，无法查询政策，请联系管理员。"

    hits = await retrieve(question, tenant_id=str(tenant_id))
    if not hits:
        return "未在公司政策文档中检索到相关内容。请换个说法描述问题，或咨询财务部门。"
    return _format_hits(hits)
