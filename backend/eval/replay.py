"""LLM 录制/回放（cassette）——把唯一"非确定+联网+花钱"的依赖冻结成可离线复跑的数据。

设计要点（面试可讲）：
  - **按有序序列回放**，不按消息内容哈希键。回放下控制流可复现 → 每个场景的 LLM 调用次数与
    顺序稳定 → 逐次 pop 即可。这样天然免疫 system prompt 里的 {today}、召回的记忆前缀、滚动摘要
    等"每次略有不同"的输入——用哈希键会频繁 miss，用序列则稳如磐石。
  - 三者接口对齐真实 ChatModel 被图/记忆/judge 用到的三种姿势：bind_tools(...).ainvoke、
    with_structured_output(Schema).ainvoke、以及直接 ainvoke。RecordingLLM 真调并落盘，
    ReplayLLM 只从录好的序列取——图对二者无感（依赖注入的红利）。

序列化：AIMessage 存 content/tool_calls/usage（token 也录下，回放仍能算成本）；
结构化输出存 model_dump()，回放时由 with_structured_output 记住的 schema 重建。
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage


# ————————————————————— 序列化 / 反序列化 —————————————————————
def _ser_ai(msg: AIMessage) -> dict[str, Any]:
    return {
        "kind": "ai",
        "content": msg.content,
        "tool_calls": [
            {"name": c["name"], "args": c["args"], "id": c.get("id")}
            for c in (msg.tool_calls or [])
        ],
        "usage": dict(msg.usage_metadata) if msg.usage_metadata else {},
    }


def _deser_ai(item: dict[str, Any]) -> AIMessage:
    usage = item.get("usage") or None
    return AIMessage(
        content=item.get("content", ""),
        tool_calls=item.get("tool_calls") or [],
        usage_metadata=usage,  # None 时不带；有则回放也能统计 token
    )


def usage_tokens(item: dict[str, Any]) -> int:
    """从一条录制项里取 total token（缺失算 0）。"""
    u = item.get("usage") or {}
    return int(u.get("total_tokens") or 0)


# ————————————————————— 录制 —————————————————————
class _RecRunnable:
    """包住真实的 bound/structured runnable：真调一次、把结果序列化进共享 sink、再原样返回。"""

    def __init__(self, real: Any, sink: list[dict], kind: str):
        self._real = real
        self._sink = sink
        self._kind = kind  # "ai" | "struct"

    async def ainvoke(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
        result = await self._real.ainvoke(messages, *args, **kwargs)
        if self._kind == "ai":
            self._sink.append(_ser_ai(result))
        else:  # 结构化输出：pydantic 实例 → dump
            self._sink.append({"kind": "struct", "data": result.model_dump()})
        return result


class RecordingLLM:
    """真跑 DeepSeek，并把**每一次** LLM 调用按顺序录进 sink（一个 list）。record/live 模式用。"""

    def __init__(self, real: Any, sink: list[dict]):
        self._real = real
        self._sink = sink

    def bind_tools(self, tools: Any, **kwargs: Any) -> _RecRunnable:
        return _RecRunnable(self._real.bind_tools(tools, **kwargs), self._sink, "ai")

    def with_structured_output(self, schema: Any, **kwargs: Any) -> _RecRunnable:
        return _RecRunnable(
            self._real.with_structured_output(schema, **kwargs), self._sink, "struct"
        )

    async def ainvoke(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
        result = await self._real.ainvoke(messages, *args, **kwargs)
        self._sink.append(_ser_ai(result))
        return result


# ————————————————————— 回放 —————————————————————
class _RepRunnable:
    def __init__(self, parent: ReplayLLM, kind: str, schema: Any = None):
        self._parent = parent
        self._kind = kind
        self._schema = schema

    async def ainvoke(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
        item = self._parent._next()
        if self._kind == "ai":
            return _deser_ai(item)
        # 结构化：用记住的 schema 把 dump 还原成 pydantic 实例
        return self._schema(**item["data"])


class ReplayLLM:
    """完全不联网：从录好的序列里逐次取。回放下控制流复现 → 取用顺序与录制时一致。"""

    def __init__(self, source: list[dict]):
        self._src = list(source)
        self._i = 0

    def _next(self) -> dict[str, Any]:
        if self._i >= len(self._src):
            # 回放耗尽：多半是图结构改了、调用次数比录制时多 → 明确报错提示重录，别静默出错
            raise RuntimeError(
                f"cassette 已耗尽（第 {self._i + 1} 次调用无录制）——图/prompt 可能变了，请 make eval-record 重录"
            )
        item = self._src[self._i]
        self._i += 1
        return item

    def bind_tools(self, tools: Any, **kwargs: Any) -> _RepRunnable:
        return _RepRunnable(self, "ai")

    def with_structured_output(self, schema: Any, **kwargs: Any) -> _RepRunnable:
        return _RepRunnable(self, "struct", schema)

    async def ainvoke(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
        return _deser_ai(self._next())
