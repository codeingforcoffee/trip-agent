import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// 前端跑在 5173，后端在 8000（跨源）—— 这里【故意不做 proxy】，让浏览器真正走跨源请求，
// 从而验证后端 M9 的动态 CORS 中间件（白名单里已含 http://localhost:5173）。
// 后端地址可用 VITE_API_BASE 覆盖，默认 http://localhost:8000（见 src/api.js）。
export default defineConfig({
  plugins: [vue()],
  server: { port: 5173 },
})
