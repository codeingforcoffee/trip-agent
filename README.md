# 差旅 Agent（trip-agent）

一个**企业级全栈** LangGraph 差旅助手,用真实业务场景串起 Agent 工程的核心能力:
DeepSeek 多轮对话、并发工具调用、Redis 分布式锁与高并发、多租户与用户身份隔离、
短期+长期记忆、上下文压缩、基础 RAG、安全模块(注入/PII/授权/HITL)、离线评测,
以及生产化收尾(SSE 流式前端、可观测与成本核算、Caddy 边缘 TLS、Docker 全栈一键起)。

> 教学式分阶段构建,里程碑 **M0 → M9 已全部完成**。每个里程碑一个 git commit,
> 交流/注释/commit 一律中文,注释偏重"为什么这么设计"。

## 架构总览

```
  Vue3 聊天 UI ──HTTPS──▶ Caddy(边缘)  ──HTTP(+X-Forwarded-Proto)──▶ FastAPI
  (登录/流式/引用/HITL)     TLS 终止 + 静态托管 + 反代                ├ 中间件:trace/成本 · 安全头 · CORS
                          (前后端同源 ⇒ 生产免 CORS)                ├ auth(JWT + refresh 轮换)
                                                                    └ /chat(SSE) ──▶ LangGraph 图
                                                                          triage → agent ↔ tools
                                                                          +compress +memory +guards +HITL
                                                                                │
             ┌──────────────────────────┬──────────────┬───────────────────────┤
        ┌────▼─────┐  ┌─────────────┐  ┌─▼───────────┐  ┌──────▼──────┐  ┌───────▼────────┐
        │ DeepSeek │  │  Postgres   │  │   Redis     │  │   Qdrant    │  │ 本地 BGE        │
        │ chat     │  │ checkpoint  │  │ 分布式锁    │  │ RAG + 语义  │  │ (fastembed/ONNX)│
        │          │  │ 记忆/业务/  │  │ 限流/幂等   │  │ 记忆(租户   │  │                 │
        │          │  │ 审计(RLS)   │  │ (t:{tid}:)  │  │ payload过滤)│  │                 │
        └──────────┘  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────┘
```

多租户隔离贯穿始终:每表带 `tenant_id`(+ Postgres RLS)、Redis key 前缀 `t:{tenant_id}:`、
Qdrant payload 按 `tenant_id` 过滤、LangGraph `thread_id = {tenant}:{user}:{conv}`;身份从 JWT
经 FastAPI deps 注入 `config.configurable`,**绝不从对话内容反推**。

## 技术栈

| 层 | 选型 |
|---|---|
| 语言/工具链 | Python 3.12 + uv |
| Agent 编排 | LangGraph + PostgresSaver(连接池版,Web 并发) |
| LLM | DeepSeek(`deepseek-chat` / `deepseek-reasoner`)· `langchain-deepseek` |
| Embedding | 本地 BGE `bge-small-zh`,经 fastembed(ONNX,不依赖 torch) |
| 向量库 | Qdrant(RAG 语料 + 语义记忆,payload 租户过滤) |
| 关系库 | Postgres(checkpoint / 长期记忆 / 业务 / 审计,RLS) |
| 缓存/锁 | Redis(分布式锁 + 限流 + 幂等 + refresh 会话族) |
| 后端 | FastAPI(SSE 流式)+ SQLAlchemy(async)+ Alembic |
| 前端 | Vue 3 + Vite(fetch 读 SSE + markdown-it 渲染;最小实现,无 Pinia/TS) |
| 边缘/部署 | Caddy(TLS 终止 + 反代 + 静态托管)+ Docker Compose 全栈 |
| 可观测 | structlog JSON 日志(trace_id / tenant_id / token 成本) |
| 评测 | 自研 harness + LLM-as-judge + cassette 录制回放(离线可复现) |

## 快速开始

前置:已安装 `docker` 与 `uv`;`cp .env.example .env` 并填入 `DEEPSEEK_API_KEY`。

### 方式一：本机开发（热重载）

```bash
make up                       # 起依赖容器 postgres / redis / qdrant
make sync                     # uv 装依赖(自动拉 Python 3.12)
make migrate && make seed     # 建表 + 灌 2 租户/用户演示数据
make ingest                   # 把样例差旅政策文档写入 Qdrant(RAG)
make run                      # 后端 :8000(--reload)
# 另开终端:
make health                   # 探活三依赖,预期 {"status":"ok",...}
cd frontend && npm install && npm run dev   # 前端 :5173
```

浏览器开 http://localhost:5173,用演示账号登录:租户 `acme` / `alice@acme.com` / `alice-pass`。

### 方式二：Docker 全栈一键起（生产形态 + 本地 HTTPS）

```bash
make up && make migrate && make seed && make ingest   # 依赖 + 数据(首次)
make stack-build              # 构建 backend + edge 镜像(backend 含依赖,首次较慢)
make stack-up                 # 起全栈(profile fullstack:依赖 + backend + Caddy 边缘)
```

浏览器开 **https://localhost**(首次需信任 Caddy 自动签发的本地 root CA)。前后端同源、无需 CORS。
`make stack-down` 停全栈。生产只需把 `Caddyfile` 里的 `localhost` 换成真实域名 → Caddy 自动走
ACME 签发/续期证书。

`make help` 查看全部命令;`make test` 跑测试(当前 130 passed)。

## 里程碑

| 里程碑 | 内容 |
|---|---|
| **M0** | 可复现地基:uv + Docker Compose + 配置中心(pydantic-settings)+ structlog + `/health` 并发探针 |
| **M1** | DeepSeek + LangGraph 最小多轮对话,PostgresSaver 做短期记忆(thread 恢复) |
| **M2** | 差旅工具集 + 并发工具调用(`asyncio.gather`)+ 错误自纠;triage 分诊/澄清门控 |
| **M3** | 多租户 + 用户身份:ORM(带 `tenant_id`)+ Alembic + RLS + JWT + deps 注入身份 |
| **M4** | Redis 分布式锁(`SET NX PX` + Lua 释放 + 看门狗续期)+ 令牌桶限流 + 下单幂等 |
| **M5** | 基础 RAG:本地 BGE + Qdrant(租户过滤)+ 带引用的政策检索 |
| **M6** | 上下文压缩(按窗口比例双水位滚动摘要)+ 长期记忆(召回注入 + 抽取去重写入) |
| **M7** | 安全:注入启发式 + PII 脱敏 + 工具 scope 授权 + 高危动作 HITL(`interrupt`)+ 审计 |
| **M8** | 离线评测 harness:黄金数据集 + 确定性指标 + LLM-as-judge + cassette 回放 |
| **M9** | 全栈打通 + 生产化:SSE 流式 UI(a)· refresh 令牌轮换(b)· Vue3 前端(c)· 可观测/成本 + 安全头 + Docker/Caddy(d) |

完整路线图(技术选型与各里程碑验证方式)见 `/Users/garry/.claude/plans/` 下的计划文件。

## 目录结构

```
trip-agent/
├── docker-compose.yml        # 依赖(postgres/redis/qdrant) + 全栈(backend/edge,profile fullstack)
├── Makefile · .env.example
├── backend/
│   ├── Dockerfile · pyproject.toml(锁 3.12)
│   ├── app/
│   │   ├── main.py           # FastAPI 入口 + lifespan(建池) + 中间件链 + /health
│   │   ├── core/             # config / logging / security / deps / observability / security_headers / dynamic_config
│   │   ├── db/               # session(async 引擎/RLS) / models / migrations
│   │   ├── infra/            # redis_client / locks / ratelimit / refresh_store / qdrant
│   │   ├── llm/              # deepseek / embeddings(BGE·fastembed)
│   │   ├── agent/            # graph / state / nodes / checkpointer / memory / context / triage / tools/
│   │   ├── rag/              # ingest / retriever(Qdrant,租户过滤)
│   │   ├── security/         # guards(注入/PII) / authz(工具 scope)
│   │   └── api/              # auth(JWT+refresh) / chat(SSE+HITL) / conversations
│   └── eval/                 # datasets / metrics / judge / run_eval
└── frontend/                 # Vue 3 + Vite 流式聊天 UI + Dockerfile + Caddyfile(边缘)
```

## 工程亮点（面试向）

- **SSE 流式成本核算**:用**纯 ASGI 中间件**而非 `BaseHTTPMiddleware`——后者 `call_next` 在 SSE
  body 发送前返回,拿不到流结束后的 token 总数;用 `ContextVar` 可变账本在同一 asyncio Context
  内从流生成器回传用量,末尾记 `http.request`(时延 + in/out tokens + `cost_cny`)。
- **`--proxy-headers` 信任边界**:TLS 在 Caddy 终止、后端跑明文,靠 `X-Forwarded-Proto` 识别 https;
  转发头可伪造,故 `--forwarded-allow-ips` 收窄到反代、后端不对外发布端口。HSTS 仅 https 下发即验证此边界。
- **单 origin 免 CORS**:前后端同源(Caddy 静态 + 反代),生产不发跨源请求;动态 CORS 退化为 dev 兜底。
- **refresh 令牌轮换 + 重放检测**:短命无状态 access + 长命有状态 refresh;Redis 会话族 + Lua 原子
  CAS 做旋转/撤销,旧 jti 重放即撤销整族(OWASP)。
- **分布式锁正确性**:原子获取(`NX PX`)+ 唯一 token + Lua 比对释放(防误删)+ 看门狗续期。
- **纵深防御**:硬边界是 authz scope + Postgres RLS;注入检测/PII 脱敏是概率性防线(可切开关)。
