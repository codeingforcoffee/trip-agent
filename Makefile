# 统一命令入口：把"怎么跑这个项目"的知识固化进 Makefile，新人 `make help` 即可上手。
# 注意：Makefile 的命令必须用 Tab 缩进（不是空格）。

.DEFAULT_GOAL := help
COMPOSE := docker compose

.PHONY: help up down ps logs sync run health test migrate revision seed ingest lock-demo eval clean

help: ## 显示所有可用命令
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

up: ## 启动依赖容器（postgres / redis / qdrant）
	$(COMPOSE) up -d

down: ## 停止并移除容器（保留数据卷）
	$(COMPOSE) down

ps: ## 查看容器状态
	$(COMPOSE) ps

logs: ## 跟踪容器日志
	$(COMPOSE) logs -f

sync: ## 安装/同步后端依赖（含 dev）
	cd backend && uv sync --group dev

run: ## 启动后端 API（开发模式，热重载，:8000）
	cd backend && uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

chat: ## 启动差旅 Agent CLI 交互模式（M1）；可加 THREAD=xxx
	cd backend && uv run python -m app.agent.cli --thread $(or $(THREAD),demo)

health: ## 探活后端（需 run 已启动）
	@curl -s http://localhost:8000/health | python3 -m json.tool

test: ## 运行测试
	cd backend && uv run pytest

migrate: ## 数据库迁移到最新（M3 引入）
	cd backend && uv run alembic upgrade head

revision: ## 生成迁移（自动对比模型）：make revision M="描述"
	cd backend && uv run alembic revision --autogenerate -m "$(or $(M),change)"

seed: ## 灌入演示数据：2 租户 + 用户（M3）
	cd backend && uv run python scripts/seed.py

ingest: ## 把样例政策文档灌入 Qdrant（M5 RAG，需先 seed + 起 qdrant）
	cd backend && uv run python -m app.rag.ingest

lock-demo: ## 分布式锁并发演示：互斥/临界区/TTL/看门狗（M4，需 Redis）
	cd backend && uv run python scripts/lock_demo.py

eval: ## 离线评测（M8 引入）
	cd backend && uv run python -m eval.run_eval

clean: ## 停止容器并删除数据卷（清空所有数据！）
	$(COMPOSE) down -v
