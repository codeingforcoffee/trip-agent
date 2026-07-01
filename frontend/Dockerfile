# 前端 + 边缘一体镜像（M9d）：构建上下文为 frontend/（见 docker-compose 的 context: ./frontend）。
#
# stage1(build) 用 node 把 Vue 编成静态产物；stage2(edge) 用 Caddy 托管静态 + 反代后端 + 终止 TLS。
# 【单 origin 的红利】：Caddy 在 / 提供前端、把 /auth /chat 等反代到 backend → 前后端同源，
#   生产环境浏览器【无需 CORS】（M9 的动态 CORS 退化为本地 dev 跨源时的兜底）。

FROM node:22-alpine AS build
WORKDIR /app
# 先装依赖（package-lock 决定，可缓存），再拷源码构建
COPY package.json package-lock.json ./
RUN npm ci
COPY . ./
# 同源部署：VITE_API_BASE 置空 → 前端用【相对路径】调 API（见 api.js 用 ?? 而非 || 处理空串）
ENV VITE_API_BASE=""
RUN npm run build

FROM caddy:2-alpine AS edge
# Caddyfile：TLS(本地 internal CA / 生产 ACME) + 静态托管 + 反代 backend
COPY Caddyfile /etc/caddy/Caddyfile
# 前端产物放到 Caddy 的站点根
COPY --from=build /app/dist /srv
EXPOSE 80 443
