"""配置中心（12-factor）。

所有可变项（连接串、密钥、模型名…）一律走环境变量，代码里零硬编码。
pydantic-settings 会：
  1. 按字段名（大小写不敏感）从环境变量 / .env 文件读取；
  2. 做类型校验（端口必须是 int、debug 必须是 bool …）；
  3. 缺失时用这里给的默认值。

这样换机器、换环境只改 .env，代码一行不动 —— 这就是"可预测"的地基。
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # env_file 给了两个路径：
    #   - "../.env"：从 backend/ 目录运行时，根目录的 .env（与 docker-compose 共用同一份）
    #   - ".env"：从仓库根目录运行时
    # 谁先命中用谁，保证无论在哪个目录启动都能读到配置。
    model_config = SettingsConfigDict(
        env_file=("../.env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",  # .env 里有多余变量（如给 compose 用的）不报错
    )

    # —— 应用元信息 ——
    app_name: str = "trip-agent"
    app_env: str = "dev"  # dev / prod
    debug: bool = True
    log_level: str = "INFO"
    log_json: bool = True  # 生产用 JSON；本地想看人类可读可设 false

    # —— Postgres（短期记忆 checkpoint + 多租户业务 + 审计）——
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "trip"
    postgres_password: str = "trip_pass"
    postgres_db: str = "trip_agent"

    # —— Postgres 连接池（SQLAlchemy 异步引擎，全部可配）——
    db_pool_size: int = 10  # 常驻连接数
    db_max_overflow: int = 20  # 峰值临时溢出连接数（池满时最多再临时开这么多）
    db_pool_timeout: int = 30  # 池满时获取连接的最长等待秒数，超时即报错（快速失败）
    db_pool_recycle: int = 1800  # 连接最大存活秒数，超过则回收重建（防被 DB/防火墙掐断）
    db_pool_pre_ping: bool = True  # 借出前 ping 一下，自动剔除失效连接
    db_echo: bool = False  # 是否打印 SQL（调试用，生产关）

    # —— Redis（分布式锁 / 限流 / 缓存）——
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_max_connections: int = 50  # 连接池上限
    redis_socket_timeout: float = 3.0  # 读写超时（秒）
    redis_socket_connect_timeout: float = 3.0  # 建连超时（秒）
    redis_health_check_interval: int = 30  # 空闲连接每隔 N 秒自检，剔除死连接

    # —— Qdrant（RAG 向量 + 语义记忆）——
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # —— DeepSeek（M1 接入）——
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"

    # —— 工具层（M2）——
    # mock 工具的模拟网络延迟（毫秒）。默认 0：测试/评测保持瞬时、时间确定。
    # 演示并发时设成 300 左右，就能在日志里看到「3 个工具并发只花 ~300ms 而非 900ms」。
    tool_mock_latency_ms: int = 300

    # —— 分诊/澄清（M2+）——
    # 是否启用 triage 节点（意图分类 + 槽位门控 + 缺槽时主动反问）。
    # 关掉则退回直连图、每轮省一次 LLM 调用，但不再主动澄清（靠 system prompt 自觉）。
    enable_triage: bool = True

    # —— JWT 鉴权（M3 引入）——
    jwt_secret: str = "dev-secret-change-me"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60

    # —— 下面这些"连接串"是由上面的零件拼出来的（@property），
    #     避免在 .env 里重复写主机/端口，单一事实来源。——

    @property
    def database_url(self) -> str:
        """SQLAlchemy 异步引擎用（M3+），驱动 asyncpg。"""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def database_dsn(self) -> str:
        """裸 DSN，给 asyncpg.connect / LangGraph PostgresSaver / Alembic 用。"""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def qdrant_url(self) -> str:
        return f"http://{self.qdrant_host}:{self.qdrant_port}"


@lru_cache
def get_settings() -> Settings:
    """缓存单例：整个进程只解析一次 .env。

    用函数而非模块级全局，是为了方便测试时用 get_settings.cache_clear() 重置。
    """
    return Settings()


# 便捷别名：大多数地方直接 `from app.core.config import settings`
settings = get_settings()
