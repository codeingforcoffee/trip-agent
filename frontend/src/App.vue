<script setup>
import { computed } from 'vue'
import { auth } from './auth.js'
import LoginView from './components/LoginView.vue'
import ChatView from './components/ChatView.vue'

// auth 是 reactive，access 有值即已登录 → 决定显示登录页还是聊天页
const loggedIn = computed(() => !!auth.access)
</script>

<template>
  <LoginView v-if="!loggedIn" />
  <ChatView v-else />
</template>
