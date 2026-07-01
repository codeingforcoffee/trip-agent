import { reactive } from 'vue'

// 简易身份状态（不上 Pinia，最小依赖）。令牌存 localStorage：演示够用。
// ⚠️ 生产注意：access 放内存/短存无妨，但 refresh 令牌落 localStorage 会被 XSS 读取——
//    生产应把 refresh 放【httpOnly + Secure + SameSite】的 cookie（前端拿不到、JS 偷不走），
//    并配 CSRF 防护。这里为演示简化。
const KEY = 'trip-agent-auth'

function load() {
  try {
    return JSON.parse(localStorage.getItem(KEY)) || {}
  } catch {
    return {}
  }
}

const saved = load()
export const auth = reactive({
  access: saved.access || '',
  refresh: saved.refresh || '',
  email: saved.email || '',
  tenant: saved.tenant || '',
})

function persist() {
  localStorage.setItem(
    KEY,
    JSON.stringify({ access: auth.access, refresh: auth.refresh, email: auth.email, tenant: auth.tenant }),
  )
}

export function setSession({ access, refresh, email, tenant }) {
  auth.access = access
  auth.refresh = refresh
  if (email !== undefined) auth.email = email
  if (tenant !== undefined) auth.tenant = tenant
  persist()
}

export function clearSession() {
  auth.access = ''
  auth.refresh = ''
  auth.email = ''
  auth.tenant = ''
  localStorage.removeItem(KEY)
}

export const isLoggedIn = () => !!auth.access
