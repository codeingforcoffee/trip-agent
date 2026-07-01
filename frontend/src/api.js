import { auth, setSession, clearSession } from './auth.js'

// 后端地址（跨源，走 M9 动态 CORS）。可用 .env 的 VITE_API_BASE 覆盖。
const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

// ————————————————————————— 认证 —————————————————————————

export async function login(tenantSlug, email, password) {
  const r = await fetch(`${API_BASE}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ tenant_slug: tenantSlug, email, password }),
  })
  if (!r.ok) throw new Error('登录失败：租户、账号或密码不正确')
  const d = await r.json()
  setSession({ access: d.access_token, refresh: d.refresh_token, email, tenant: tenantSlug })
  return d
}

export async function logout() {
  // 通知后端撤销整条会话家族（幂等）；无论成败都清本地
  if (auth.refresh) {
    try {
      await fetch(`${API_BASE}/auth/logout`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ refresh_token: auth.refresh }),
      })
    } catch {
      /* 忽略网络错误 */
    }
  }
  clearSession()
}

// 用 refresh 令牌换新 access（轮换：后端会返回新的 refresh，旧的立即失效）。
// 返回 true=刷新成功；false=refresh 也失效（需重新登录）。
async function tryRefresh() {
  if (!auth.refresh) return false
  const r = await fetch(`${API_BASE}/auth/refresh`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ refresh_token: auth.refresh }),
  })
  if (!r.ok) {
    clearSession()
    return false
  }
  const d = await r.json()
  setSession({ access: d.access_token, refresh: d.refresh_token })
  return true
}

// 带鉴权的 POST；遇 401 自动刷新一次再重试（refresh 也失效则抛错）。
async function authedPost(path, payload) {
  const call = () =>
    fetch(`${API_BASE}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${auth.access}` },
      body: JSON.stringify(payload),
    })
  let r = await call()
  if (r.status === 401) {
    if (await tryRefresh()) r = await call()
  }
  return r
}

// ————————————————————————— SSE 读流 —————————————————————————

// 逐个产出 {event, data} —— 手写解析 SSE 线格式：event: <类型> / data: <JSON> / 空行分隔。
// 用 fetch + ReadableStream（而非原生 EventSource）才能 POST + 带 Authorization 头。
async function* readSSE(response) {
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    let sep
    while ((sep = buffer.indexOf('\n\n')) !== -1) {
      const frame = buffer.slice(0, sep)
      buffer = buffer.slice(sep + 2)
      let event = 'message'
      let data = ''
      for (const line of frame.split('\n')) {
        if (line.startsWith('event:')) event = line.slice(6).trim()
        else if (line.startsWith('data:')) data += line.slice(5).trim()
      }
      yield { event, data: data ? JSON.parse(data) : {} }
    }
  }
}

// 把一次 POST（/chat 或 /chat/resume）的 SSE 事件分发给回调。
async function streamPost(path, payload, handlers) {
  const r = await authedPost(path, payload)
  if (r.status === 401) {
    clearSession()
    handlers.error?.({ message: '登录已过期，请重新登录' })
    return
  }
  if (!r.ok || !r.body) {
    handlers.error?.({ message: `请求失败（${r.status}）` })
    return
  }
  // 按事件名分发。注意后端出错会发 event:error（如图崩溃），必须有对应 handler，
  // 否则错误被静默吞掉、界面只剩空白气泡（这正是之前"预定返回空白"的表层原因）。
  for await (const { event, data } of readSSE(r)) {
    handlers[event]?.(data)
  }
}

// 发起一轮对话。handlers: {token, tool_call, tool_result, citation, interrupt, usage, done, onError}
export function chatStream(message, conversationId, handlers) {
  return streamPost('/chat', { message, conversation_id: conversationId }, handlers)
}

// HITL 续跑：带上用户对高危动作的批准/拒绝。
export function resumeStream(conversationId, approved, handlers) {
  return streamPost('/chat/resume', { conversation_id: conversationId, approved }, handlers)
}
