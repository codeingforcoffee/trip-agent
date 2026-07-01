# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目性质（先读这一段）

这是一个**教学式分阶段构建**的企业级差旅 Agent(用户为准备 AI Agent 工程师面试而建)。开发按里程碑 **M0 → M9** 推进:
- **每个里程碑 = 概念讲解 → 带详尽中文注释的代码 → 跑起来验证 → 面试话术 → 一个 git commit**(commit message 用 `M{n}: ...` / `M{n}+: ...` 格式)。
- 代码注释、commit、与用户的交流**一律中文**;注释偏重"为什么这么设计",不只是"做了什么"。
- 完整路线图(技术选型、各里程碑产物与验证方式)在计划文件:`/Users/garry/.claude/plans/redis-langgraph-deepseek-agent-agent-ra-jiggly-bengio.md`。**接手前先读它**,了解当前进度与下一步。
- **进度:M0 → M9 已全部完成**(M9d 生产化收尾:可观测中间件 + 安全头 + Docker 全栈 + Caddy 边缘 TLS)。后续为可选精进(语义缓存安全版、记忆更新智能闸门、M5+ 混合检索+重排)或面试演练;新增里程碑仍照上面的交付节奏与 commit 格式。当前状态始终以 `git log` 为准。

## 常用命令

根目录有 `Makefile` 统一入口(`make help` 看全部)。**所有 `uv`/`pytest` 命令的工作目录是 `backend/`**(uv 项目根在那里);Make 目标会自动 `cd backend`。

```bash
make up        # 起依赖容器 postgres/redis/qdrant(docker compose)
make sync      # uv sync --group dev(自动拉 Python 3.12 + 依赖)
make run       # uvicorn app.main:app --reload :8000
make health    # curl /health,确认三依赖 ok
make test      # pytest
make down      # 停容器(留数据卷);make clean 连数据卷一起删
make stack-build  # 构建全栈镜像(backend 含依赖,首次慢)
make stack-up     # 全栈一键起(profile fullstack:依赖+backend+edge/Caddy);浏览器开 https://localhost
make stack-down   # 停全栈
```

**两种跑法**:①本机直跑(`make up` 起依赖 + `make run` 起后端 + 前端 `npm run dev`,`*_HOST=localhost`);②全栈容器(`make stack-up`,backend/edge 在 compose 网络里,`*_HOST` 由 compose 覆盖为服务名)。两者共用同一份 `.env` 互不打架。`backend`/`edge` 归入 compose `profile: fullstack`,所以 `make up` 只起三个依赖、**不会误触发 backend 重镜像构建**。

单测 / 直接用 uv(注意必须在 `backend/` 下):
```bash
cd backend && uv run pytest tests/test_smoke.py::test_health_contract -q
cd backend && uv run pytest -k health -q
cd backend && uv run ruff check . && uv run ruff format .
```

依赖未起时本机没装 `docker` daemon 要先 `open -a Docker`。`postgres` 镜像用的是 **`16-alpine`**(本机已缓存,启动快);`qdrant-client` 钉在 **`<1.13`** 以匹配服务端 `v1.12.4`,改动别破坏这个对齐。Python 锁 **3.12**(系统默认是 3.14,过新,库未跟进)。

## 架构要点(big picture)

**配置驱动 + 池化 + 健康探针** 是贯穿全项目的三条主线,M0 已落地,后续里程碑都在此之上扩展。

- **配置中心 `app/core/config.py`**:唯一的 `Settings`(pydantic-settings)。所有可变项(连接、池参数、密钥、模型名)走环境变量,**代码零硬编码**。连接串(`database_url`/`database_dsn`/`redis_url`/`qdrant_url`)用 `@property` 由主机/端口等零件**拼出**,单一事实来源——加配置项就加字段,不要在别处拼串。`settings` 是 `@lru_cache` 单例。
- **`.env` 在仓库根目录**,被 docker-compose 和后端**共用同一份**。后端通过 `env_file=("../.env", ".env")` 读取,所以无论从 `backend/` 还是仓库根启动都能读到。`.env` 已 gitignore,`.env.example` 是模板。
- **连接池在 lifespan 里全局建一次**(`app/main.py` 的 `lifespan`),存进 `app.state`(`db_engine`/`db_sessionmaker`/`redis`/`redis_pool`/`qdrant`),所有请求复用,关闭时 `dispose`/`aclose`。新增需要连接的代码**从 `app.state` 取池**,不要自己新建连接。
  - Postgres:`app/db/session.py` 的 SQLAlchemy 异步引擎(池参数全走 config);`get_session` 是给 M3+ 路由用的 `Depends` 依赖。
  - Redis:`app/infra/redis_client.py` 的显式 `ConnectionPool`(M4 的锁/限流从这借连接)。
- **`/health` 是"环境就绪"的唯一事实来源**:`asyncio.gather` 并发探活三依赖,各自 try/except,任一挂了返回 `status=degraded`(HTTP 仍 200,读 body 判断),并暴露连接池实时状态。客户端/引擎都是**懒连接**,依赖没起也不阻断应用启动——这是刻意的"降级不崩溃"设计。
- **结构化日志 `app/core/logging.py`**:structlog 输出 JSON(`LOG_JSON`),`merge_contextvars` 让请求入口 bind 的 `trace_id`/`tenant_id` 自动带进每条日志。用 `get_logger(__name__)`。绑定在 **M9d 真正落地**:`ObservabilityMiddleware` 入口清并绑 `trace_id`,`deps.get_identity` 鉴权后补绑 `tenant_id`/`user_id`。
- **中间件链(M9d,`main.py` 由外到内:Observability → SecurityHeaders → DynamicCORS)**:两个自研中间件都是**纯 ASGI**而非 `BaseHTTPMiddleware`——因为 SSE 流式响应下 `BaseHTTPMiddleware` 的 `call_next` 会在 body 发送前返回,拿不到"流结束后的 token 总数"、也测不准整条流时延。token 用量靠 `observability._usage_var`(ContextVar **可变账本**)从 `/chat` 流生成器 `record_usage()` 回传给最外层中间件,末尾记一条 `http.request`(时延 + in/out/total tokens + `cost_cny`)。HSTS 只在 https 下发(读 `scope['scheme']`,由 uvicorn `--proxy-headers` 依 `X-Forwarded-Proto` 置位——这就是**信任边界**:`--forwarded-allow-ips` 必须收窄到反代)。
- **全栈/边缘拓扑(M9d)**:生产由 **Caddy** 边缘终止 TLS(本地内置 CA、生产改域名即 ACME)+ 托管前端静态 + 反代 `/auth /chat /conversations /health` → backend。前后端**同源 ⇒ 生产免 CORS**(`dynamic_config` 的动态 CORS 退化为 dev 兜底)。`app.state` 在 M9 另新增 `checkpointer_pool`(连接池版 `AsyncPostgresSaver`——Web 并发不能共用单连接)与懒建缓存的 `agent_graph`。

## 目录与里程碑映射

`backend/app/` 下按层组织,每层对应里程碑(尚未到的目录还不存在,新建时遵循同样的"配置驱动 + 从 app.state 取池"约定):

| 目录 | 职责 | 里程碑 |
|---|---|---|
| `core/` | config / logging / security(JWT+refresh) / deps / observability(trace+成本,纯ASGI) / security_headers(HSTS 等) / dynamic_config(热更新 CORS,可选 Apollo) | M0、M3、M9 |
| `db/` | 连接池 session / ORM models / Alembic 迁移 | M0、M3 |
| `infra/` | redis_client / 分布式锁 locks / 限流 ratelimit / qdrant | M0、M4、M5 |
| `llm/` | deepseek(ChatDeepSeek) / embeddings(本地 BGE) | M1、M5 |
| `agent/` | LangGraph graph/state/nodes、PostgresSaver checkpointer、memory、context 压缩、tools/ | M1/M2/M6 |
| `rag/` | ingest / retriever(Qdrant,payload 带 tenant_id 过滤) | M5 |
| `security/` | guards(注入/PII) / authz(工具 scope) | M7 |
| `api/` | auth(登录 + refresh 轮换/撤销) / chat(SSE 流式 + HITL resume) / conversations | M3、M9 |
| `eval/`(backend 下) | 离线评测:datasets / metrics / judge / run_eval | M8 |
| `frontend/`(仓库根) | Vue 3 + Vite 最小流式聊天 UI(fetch 读 SSE + markdown-it 渲染;**无 Pinia/TS**)+ `Dockerfile`/`Caddyfile`(边缘) | M9 |

**多租户隔离贯穿始终**(M3 起):每张表带 `tenant_id`,Redis key 前缀 `t:{tenant_id}:`,Qdrant payload 按 `tenant_id` 过滤,LangGraph `thread_id = {tenant_id}:{user_id}:{conv_id}`,身份经 JWT → FastAPI deps → 注入 LangGraph `config.configurable`。新增任何持久化/缓存/检索代码都必须带上租户维度。

## 技术栈

Python 3.12 + uv · FastAPI(SSE)+ SQLAlchemy(async)+ Alembic · LangGraph + PostgresSaver · DeepSeek(`langchain-deepseek`)· 本地 BGE embedding(fastembed/ONNX,不依赖 torch)· Qdrant(向量)· Postgres(checkpoint/记忆/业务/审计)· Redis(锁/限流)· Vue 3 前端(fetch+SSE+markdown-it)· Caddy 边缘 TLS + Docker 全栈(compose profile `fullstack`)· structlog 日志(trace_id/tenant_id/成本)。
