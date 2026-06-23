# 差旅 Agent（trip-agent）

一个**企业级全栈** LangGraph 差旅助手,用真实的业务场景串起 Agent 工程的核心能力:
DeepSeek 多轮对话、并发工具调用、Redis 分布式锁与高并发、多租户与用户身份隔离、
短期+长期记忆、上下文压缩、基础 RAG、安全模块、离线评测。

> 教学式分阶段构建,详见里程碑 M0–M9(见 `/Users/garry/.claude/plans/` 下的计划文件)。

## 技术栈

| 层 | 选型 |
|---|---|
| 语言/工具链 | Python 3.12 + uv |
| Agent 编排 | LangGraph + PostgresSaver |
| LLM | DeepSeek(`deepseek-chat` / `deepseek-reasoner`) |
| Embedding | 本地 BGE(sentence-transformers) |
| 向量库 | Qdrant |
| 关系库 | Postgres |
| 缓存/锁 | Redis |
| 后端 | FastAPI(SSE 流式)+ SQLAlchemy + Alembic |
| 前端 | Vue 3 + Vite + TS + Pinia(M9) |

## 快速开始(M0)

前置:已安装 `docker` 与 `uv`。

```bash
# 1. 准备环境变量(已带 dev 默认值,可直接用)
cp .env.example .env        # 若尚未创建

# 2. 启动依赖容器(postgres / redis / qdrant)
make up

# 3. 安装后端依赖(uv 会自动拉取 Python 3.12)
make sync

# 4. 启动后端
make run

# 5. 另开一个终端,探活三个依赖
make health
# 预期: {"status": "ok", "deps": {"postgres": {"ok": true}, ...}}
```

`make help` 查看所有命令。

## 目录结构

```
trip-agent/
├── docker-compose.yml      # postgres / redis / qdrant
├── Makefile                # 统一命令入口
├── .env.example            # 环境变量模板
└── backend/
    ├── pyproject.toml      # uv 项目(锁 Python 3.12)
    └── app/
        ├── main.py         # FastAPI 入口 + /health
        └── core/           # config(配置中心) / logging(structlog)
```

后续里程碑会逐步加入 `agent/`(LangGraph 图)、`infra/`(锁/限流)、`rag/`、`security/`、`eval/`、`frontend/`。
