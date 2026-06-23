"""DeepSeek 对话模型接入。

DeepSeek 提供 **OpenAI 兼容** 的接口，官方有 langchain-deepseek 集成（ChatDeepSeek，
底层是 ChatOpenAI，但额外处理了 deepseek-reasoner 的 reasoning_content）。
我们把 api_key / base_url / model 都从 config 注入（来自根目录 .env），代码零硬编码。

两个模型：
  - deepseek-chat     （V3）：通用对话 + 工具调用，差旅主流程用它；
  - deepseek-reasoner （R1）：强推理，留给 M6 的规划 / M8 的判分等需要"想清楚"的场景。

注意 key 来自 pydantic 读的 .env 文件，不是进程环境变量，所以必须显式传给 ChatDeepSeek
（它默认只读 os.environ['DEEPSEEK_API_KEY']）。
"""

from __future__ import annotations

from functools import lru_cache

from langchain_deepseek import ChatDeepSeek

from app.core.config import settings


@lru_cache(maxsize=8)
def get_llm(model: str | None = None, temperature: float = 0.3) -> ChatDeepSeek:
    """构造（并缓存）一个 DeepSeek 聊天模型。

    构造本身不发起网络请求（懒连接），首次 ainvoke 时才真正调用 API。
    """
    if not settings.deepseek_api_key:
        raise RuntimeError(
            "未配置 DEEPSEEK_API_KEY。请在根目录 .env 填入你的 DeepSeek API Key "
            "（申请：https://platform.deepseek.com）。"
        )
    return ChatDeepSeek(
        model=model or settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        api_base=settings.deepseek_base_url,
        temperature=temperature,
        timeout=60,
        max_retries=2,  # 网络抖动自动重试
    )
