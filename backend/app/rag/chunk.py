"""结构感知切块（M5 写路径的第一步，也是召回天花板所在）。

为什么不按定长字符瞎切：那会把"住宿费上限 600 元/晚"从中间劈开，检索到半句话。
政策文档天生有层级标题（章/节/条），最优策略是**沿标题边界切，一个叶子小节 = 一块**；
小节过长再按段落递归切并保留**重叠**（防边界句被腰斩）。

两个面试加分点：
  1. **标题路径前缀**：每块文本前面拼上它的标题路径
     （如「Acme 差旅报销管理办法 > 住宿标准 > 一线城市：」）。这样即使检索到孤立块，
     LLM 也知道它"讲的是什么"，引用也天然有出处——这叫 contextual chunking / 标题增强。
  2. **重叠（overlap）**：相邻块共享尾部若干字符，避免被切点正好劈开的句子两边都召回不全。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# 匹配 Markdown ATX 标题：# / ## / ### ...，捕获井号数（层级）与标题文本
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass(frozen=True)
class Chunk:
    """一个待入库的文档块。text 已带标题路径前缀；source/section 用于检索后生成引用。"""

    text: str  # 入库与拼上下文用的正文（含标题路径前缀）
    source: str  # 来源文档名（如「Acme 差旅报销管理办法」）
    section: str  # 标题路径（如「住宿标准 > 一线城市」）


def _split_long(body: str, max_chars: int, overlap: int) -> list[str]:
    """把过长的小节正文按段落聚合到 max_chars 以内，相邻片段保留 overlap 重叠。

    优先沿空行（段落）切，尽量不破坏语义；单段就超长时再按字符硬切兜底。
    """
    if len(body) <= max_chars:
        return [body]
    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    pieces: list[str] = []
    buf = ""
    for p in paras:
        # 单段本身就超长：先冲掉缓冲，再把这段按字符硬切
        if len(p) > max_chars:
            if buf:
                pieces.append(buf)
                buf = ""
            for i in range(0, len(p), max_chars):
                pieces.append(p[i : i + max_chars])
            continue
        if buf and len(buf) + 1 + len(p) > max_chars:
            pieces.append(buf)
            # 新片段带上一片段的尾部 overlap 个字符，保证跨切点的上下文不丢
            tail = pieces[-1][-overlap:] if overlap else ""
            buf = (tail + "\n" + p) if tail else p
        else:
            buf = (buf + "\n" + p) if buf else p
    if buf:
        pieces.append(buf)
    return pieces


def chunk_markdown(text: str, *, source: str, max_chars: int, overlap: int) -> list[Chunk]:
    """把一篇 Markdown 政策文档切成带标题路径的块。

    维护一个"标题栈"：遇到 `## 住宿标准` 就把第 2 层设为它并截断更深层级；正文累积在
    当前标题路径下，遇到新标题就把累积的小节冲刷成块。叶子小节过长则递归切 + 重叠。
    """
    lines = text.splitlines()
    heading_stack: list[str] = []  # 下标 0 = h1，1 = h2 ...
    body_lines: list[str] = []
    chunks: list[Chunk] = []

    def flush() -> None:
        body = "\n".join(body_lines).strip()
        body_lines.clear()
        if not body:
            return
        # 标题路径：跳过 h1（文档标题，已作为 source），用 h2 及更深层级拼
        section = " > ".join(h for h in heading_stack[1:] if h)
        prefix = f"{source}" + (f" > {section}" if section else "") + "："
        for piece in _split_long(body, max_chars, overlap):
            chunks.append(Chunk(text=f"{prefix}\n{piece}", source=source, section=section))

    for line in lines:
        m = _HEADING.match(line)
        if m:
            flush()  # 进入新标题前，先把上一小节的正文冲刷掉
            level = len(m.group(1))
            title = m.group(2).strip()
            # 截断到 level-1 层，再把本层设为 title（更深层级随之失效）
            del heading_stack[level - 1 :]
            while len(heading_stack) < level - 1:
                heading_stack.append("")  # 跳级标题（如直接出现 h3）时占位
            heading_stack.append(title)
        else:
            body_lines.append(line)
    flush()  # 文件末尾的最后一节
    return chunks
