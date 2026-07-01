<script setup>
import { ref } from 'vue'
import { login } from '../api.js'

// 预填演示账号（与 scripts/seed.py 一致）：acme/alice-pass 有下单权限，globex/bob-pass 仅对话。
const tenant = ref('acme')
const email = ref('alice@acme.com')
const password = ref('alice-pass')
const err = ref('')
const busy = ref(false)

async function submit() {
  err.value = ''
  busy.value = true
  try {
    await login(tenant.value.trim(), email.value.trim(), password.value)
  } catch (e) {
    err.value = e.message
  } finally {
    busy.value = false
  }
}
</script>

<template>
  <div class="login-wrap">
    <h1>差旅 Agent 登录</h1>
    <input v-model="tenant" placeholder="租户 slug（acme / globex）" />
    <input v-model="email" placeholder="邮箱" />
    <input v-model="password" type="password" placeholder="密码" @keyup.enter="submit" />
    <div v-if="err" class="err">{{ err }}</div>
    <button class="btn-primary" style="width: 100%" :disabled="busy" @click="submit">
      {{ busy ? '登录中…' : '登录' }}
    </button>
  </div>
</template>
