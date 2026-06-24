"""差旅/报销政策查询工具。

M2 先用**内置关键词规则**返回政策条目（确定性、离线、够用）；
M5 会把这个工具的实现**原地替换**为基于 Qdrant 的真实 RAG 检索（带租户过滤 + 引用），
而工具的对外签名/名字保持不变——这正是"工具是稳定接口、实现可演进"的好例子。
"""

from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.agent.tools._mock import simulate_latency

# 关键词命中 -> 政策原文。M5 换成向量检索后，这张表会变成 Qdrant 里的文档切块。
_POLICY: list[tuple[tuple[str, ...], str]] = [
    (
        ("住宿", "酒店", "房费", "住宿费"),
        "住宿费报销上限：一线城市(北上广深) 600 元/晚，二线城市 450 元/晚，"
        "其他城市 350 元/晚。须提供住宿发票，超标部分自理。",
    ),
    (
        ("机票", "航班", "飞机", "经济舱", "公务舱"),
        "机票报销：经济舱据实全额报销；公务舱仅限总监及以上职级。须提供登机牌 + 电子行程单。",
    ),
    (
        ("高铁", "火车", "动车", "二等座", "一等座"),
        "铁路报销：二等座据实全额报销；一等座需部门负责人事前审批。",
    ),
    (
        ("餐", "伙食", "餐补", "餐饮"),
        "餐补标准：出差期间每日 100 元，无需发票，按实际出差天数发放。",
    ),
    (
        ("打车", "市内交通", "出租", "网约车"),
        "市内交通：每日上限 80 元，须提供行程单；机场/车站往返不计入此上限。",
    ),
]
_DEFAULT = "未匹配到对应政策条目。请细化问题（如住宿/机票/高铁/餐补/市内交通），或咨询财务部门。"


class PolicyQuery(BaseModel):
    question: str = Field(
        min_length=1, description="关于差旅或报销政策的问题，如「住宿费报销上限多少」"
    )


@tool(args_schema=PolicyQuery)
def query_travel_policy(question: str) -> str:
    """查询公司差旅与报销政策（住宿/机票/高铁/餐补/市内交通的标准与流程）。

    适用场景：用户问「能报多少」「需要什么发票」「什么职级能坐公务舱」等政策类问题。
    注意：政策类问题必须调本工具，不要凭空回答金额或规则。
    （M2 为内置规则，M5 升级为基于政策文档的 RAG 检索，会返回带引用的答案。）
    """
    simulate_latency()
    for keywords, text in _POLICY:
        if any(k in question for k in keywords):
            return text
    return _DEFAULT
