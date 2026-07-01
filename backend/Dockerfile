# 后端镜像（M9d）：uv 多阶段构建。构建上下文为 backend/（见 docker-compose 的 context: ./backend）。
#
# 为什么多阶段：
#   - builder 阶段带 uv + 编译链，把依赖装进 /app/.venv；
#   - runtime 阶段只带 venv + 源码，不带 uv/编译链 → 镜像更小、攻击面更小。
# 为什么先拷 lock 再拷源码：依赖层可被 Docker 层缓存复用——只改业务代码时不必重装依赖。
# 注：本项目 embedding 走 fastembed(ONNX) 而非 torch，镜像相对可控；若换 sentence-transformers
#     会拉入 torch，镜像可达数 GB（构建慢）——这也是本里程碑"写配置、不强跑重镜像"的原因。

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
# UV_COMPILE_BYTECODE：预编译 .pyc 让容器启动更快；UV_LINK_MODE=copy：cache 挂载下用拷贝避免符号链接失效
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy
WORKDIR /app
# 1) 先只装依赖（不装本项目），这一层随 lock 改变才失效
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev
# 2) 再拷源码并装项目本身（--frozen 保证与 lock 完全一致，可复现）
COPY . ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

FROM python:3.12-slim-bookworm AS runtime
# 非 root 运行（最小权限：即便被攻破也拿不到 root）
RUN useradd --create-home --uid 10001 appuser
WORKDIR /app
COPY --from=builder --chown=appuser:appuser /app /app
# venv 的可执行文件进 PATH；PYTHONUNBUFFERED 让日志实时刷出（容器里别缓冲）
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1
USER appuser
EXPOSE 8000
# --proxy-headers：信任反代(Caddy)传来的 X-Forwarded-Proto/For，据此把 scheme 置为 https、
#   拿到真实客户端 IP（安全头/限流/日志都依赖它）。
# --forwarded-allow-ips=*：信任【所有】来源的转发头——仅因为本容器不对外发布端口、只经 edge 访问，
#   docker 网络内可信。★生产务必收窄到反代/LB 的 IP 段★，否则外部可伪造 X-Forwarded-Proto。
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]
