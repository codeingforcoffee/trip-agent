import MarkdownIt from 'markdown-it'

// —————————————————— 为什么这样配 ——————————————————
// 我们要把 LLM 输出（不可信文本）用 v-html 塞进 DOM，天然有 XSS 风险。
// markdown-it 的默认安全姿势足以覆盖这里的场景，无需再拉 DOMPurify：
//   1) html:false —— 源码里的裸 HTML（如 <script>、<img onerror>）被「转义成文本」
//      而非当作标签渲染，从根上堵死 HTML 注入。这是防 XSS 的最关键一项。
//   2) 内置 validateLink —— 默认就拦 javascript:/vbscript:/file: 等危险协议，
//      所以 [x](javascript:alert(1)) 不会生成可点的恶意链接。
// 只有当业务确实要放行 html:true 时，才必须叠加 DOMPurify 做净化——这里不需要。
const md = new MarkdownIt({
  html: false, // 不放行原始 HTML → 天然防 XSS（最重要的一条）
  linkify: true, // 裸 URL 自动成链接，贴合聊天里贴链接的习惯
  breaks: true, // 单个换行也渲染成 <br>，符合 LLM 逐行输出的直觉
})

// 链接统一新窗口打开，并加 rel 防「reverse tabnabbing」（新页面通过 window.opener 篡改原页）。
// markdown-it 的渲染规则是可插拔的：包住原 link_open 规则，补两个属性即可。
const defaultLinkOpen =
  md.renderer.rules.link_open ||
  function (tokens, idx, options, env, self) {
    return self.renderToken(tokens, idx, options)
  }
md.renderer.rules.link_open = function (tokens, idx, options, env, self) {
  tokens[idx].attrSet('target', '_blank')
  tokens[idx].attrSet('rel', 'noopener noreferrer')
  return defaultLinkOpen(tokens, idx, options, env, self)
}

// 流式场景：token 逐个到达，msg.text 每次追加后都会重渲一遍整段 Markdown。
// 中途可能出现「未闭合的代码块/列表」→ 渲染略歪，但补齐后自动归位，无需特殊处理。
export function renderMarkdown(text) {
  return md.render(text || '')
}
