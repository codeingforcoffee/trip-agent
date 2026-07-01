<script setup>
import { reactive, ref, nextTick } from 'vue'
import { auth, clearSession } from '../auth.js'
import { chatStream, resumeStream, logout } from '../api.js'

// 一个浏览器会话固定一个会话 id → thread_id=tenant:user:conv 稳定，多轮对话有短期记忆。
const convId = (crypto.randomUUID && crypto.randomUUID()) || `web-${Date.now()}`
const messages = reactive([])
const input = ref('')
const busy = ref(false)
const msgsEl = ref(null)

function scroll() {
  nextTick(() => {
    if (msgsEl.value) msgsEl.value.scrollTop = msgsEl.value.scrollHeight
  })
}

// 把 SSE 事件映射到某条 assistant 消息上（token 打字机、工具/引用、HITL 中断、用量）。
function makeHandlers(msg) {
  return {
    token: (d) => {
      msg.text += d.text
      scroll()
    },
    tool_call: (d) => msg.tools.push(`🔧 ${d.name}(${JSON.stringify(d.args)})`),
    tool_result: () => {},
    citation: (d) => msg.citations.push(d.snippet),
    interrupt: (d) => {
      msg.interrupt = d // 后端流到此结束，等用户确认后二次请求 resume
      msg.streaming = false
      scroll()
    },
    usage: (d) => (msg.usage = d.total_tokens),
    done: () => (msg.streaming = false),
    // 事件名对齐后端的 event:error（此前叫 onError 收不到 → 空白气泡）
    error: (d) => {
      msg.error = (d && d.message) || '对话处理出错，请重试'
      msg.streaming = false
    },
  }
}

async function send() {
  const text = input.value.trim()
  if (!text || busy.value) return
  input.value = ''
  messages.push(reactive({ role: 'user', text }))
  const msg = reactive({
    role: 'assistant',
    text: '',
    tools: [],
    citations: [],
    interrupt: null,
    usage: null,
    error: '',
    streaming: true,
  })
  messages.push(msg)
  busy.value = true
  scroll()
  try {
    await chatStream(text, convId, makeHandlers(msg))
  } finally {
    busy.value = false
    msg.streaming = false
    scroll()
  }
}

// HITL：用户对高危动作（如下单）批准/拒绝 → 二次请求续跑同一会话，事件继续写回这条消息。
async function decide(msg, approved) {
  msg.interrupt = null
  msg.streaming = true
  busy.value = true
  try {
    await resumeStream(convId, approved, makeHandlers(msg))
  } finally {
    busy.value = false
    msg.streaming = false
    scroll()
  }
}

async function onLogout() {
  await logout()
  clearSession()
}
</script>

<template>
  <div class="chat">
    <header>
      <strong>差旅 Agent</strong>
      <span class="who">{{ auth.email }} @ {{ auth.tenant }}</span>
      <button class="btn-ghost" @click="onLogout">退出</button>
    </header>

    <div ref="msgsEl" class="msgs">
      <div v-for="(m, i) in messages" :key="i" class="msg" :class="m.role">
        <!-- 工具调用过程（assistant） -->
        <div v-if="m.tools && m.tools.length" class="events">
          <span v-for="(t, ti) in m.tools" :key="ti" class="event-chip">{{ t }}</span>
        </div>
        <!-- 气泡：流式时带闪烁光标 -->
        <div class="bubble" :class="{ cursor: m.streaming }">{{ m.text }}</div>
        <!-- RAG 引用 -->
        <div v-for="(c, ci) in m.citations || []" :key="ci" class="citation">📎 {{ c }}</div>
        <!-- HITL 高危动作确认 -->
        <div v-if="m.interrupt" class="hitl">
          ⚠️ {{ m.interrupt.message || '需要确认高危操作' }}
          <div v-for="(a, ai) in m.interrupt.actions || []" :key="ai">· {{ a.tool }}({{ JSON.stringify(a.args) }})</div>
          <div class="actions">
            <button class="btn-primary" @click="decide(m, true)">批准</button>
            <button class="btn-ghost" @click="decide(m, false)">拒绝</button>
          </div>
        </div>
        <div v-if="m.usage" class="events">用量 {{ m.usage }} tokens</div>
        <div v-if="m.error" class="events" style="color: #dc2626">{{ m.error }}</div>
      </div>
    </div>

    <div class="composer">
      <input
        v-model="input"
        placeholder="问点什么，例如：上海住宿报销上限是多少？"
        :disabled="busy"
        @keyup.enter="send"
      />
      <button class="btn-primary" :disabled="busy || !input.trim()" @click="send">
        {{ busy ? '…' : '发送' }}
      </button>
    </div>
  </div>
</template>
