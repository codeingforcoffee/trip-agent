"""本地 BGE 文本嵌入（M5 RAG 的"向量化"环节）。

为什么本地、为什么 fastembed：
  - 本地：离线、免费、可复现（契合"可预测"主线），且数据不出境（合规友好）。
  - fastembed（Qdrant 官方，ONNX 运行时）：不依赖 PyTorch，安装轻、冷启动快、推理省内存；
    用的仍是 `BAAI/bge-small-zh-v1.5` 同一份权重，**嵌入向量与 sentence-transformers 一致**，
    只是运行时更贴近生产。换 sentence-transformers 也只需改这一个文件。

三个面试会追问的工程细节：
  1. **单例 + 懒加载**：模型权重要读进内存，只该做一次。用 lru_cache 缓存实例，
     首次 embed 时才真正下载/加载（不用 RAG 的命令不付这份代价）。
  2. **丢进线程池**：fastembed 的 embed 是同步、CPU 密集的阻塞调用；在 async 路径里
     直接调会卡死事件循环。用 asyncio.to_thread 丢到线程池，事件循环可继续干别的。
  3. **非对称检索（query vs passage）**：BGE 这类模型对"查询"和"文档"分别有最优编码方式
     （查询侧通常加一句指令前缀）。fastembed 的 query_embed/passage_embed 已按模型封装好，
     我们对查询用 query_embed、对文档用 embed，召回更准。
"""

from __future__ import annotations

import asyncio
from functools import lru_cache

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger("app.llm.embeddings")


@lru_cache(maxsize=1)
def get_embedder():
    """进程内唯一的嵌入模型实例（懒加载 + 缓存）。

    fastembed 首次构造时会把 ONNX 权重下载到本地缓存（之后离线可用）。
    延迟到这里 import，避免不碰 RAG 的代码路径也要扛 fastembed 的导入开销。
    """
    from fastembed import TextEmbedding

    log.info("embeddings.load_model", model=settings.embedding_model)
    return TextEmbedding(model_name=settings.embedding_model)


def embedding_dim() -> int:
    """返回向量维度（512）。读模型元数据，**不触发下载**——给建集合时定 VectorParams 用。"""
    from fastembed import TextEmbedding

    for m in TextEmbedding.list_supported_models():
        if m["model"] == settings.embedding_model:
            return int(m["dim"])
    raise ValueError(f"未知嵌入模型：{settings.embedding_model}")


def _embed_documents_sync(texts: list[str]) -> list[list[float]]:
    # embed() 返回 numpy 向量的生成器；在线程内 materialize 成 list，再转 Python list[float]
    return [v.tolist() for v in get_embedder().embed(texts)]


def _embed_query_sync(text: str) -> list[float]:
    # query_embed：按模型封装好的"查询侧"编码（BGE 会加检索指令前缀），比 embed 更贴检索场景
    return next(iter(get_embedder().query_embed([text]))).tolist()


async def embed_documents(texts: list[str]) -> list[list[float]]:
    """把一批文档块向量化（写路径 / ingest 用）。CPU 密集 → 丢线程池。"""
    return await asyncio.to_thread(_embed_documents_sync, texts)


async def embed_query(text: str) -> list[float]:
    """把单条查询向量化（读路径 / retriever 用）。"""
    return await asyncio.to_thread(_embed_query_sync, text)
