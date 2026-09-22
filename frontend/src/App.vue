<script setup lang="ts">
import { computed, onMounted } from 'vue'
import { RouterView, useRoute } from 'vue-router'

import AppHeader from '@/components/AppHeader.vue'
import { useAnalyzeStore } from '@/stores/analyze'

const store = useAnalyzeStore()
const route = useRoute()

onMounted(() => void store.init())

// 离开首页时不要继续占用轮询/SSE（结果页自己会重新订阅）
const isHome = computed(() => route.name === 'home')
</script>

<template>
  <div class="app-shell">
    <AppHeader :is-home="isHome" />

    <main class="app-main">
      <div v-if="store.bootError" class="alert alert-err" style="margin-bottom: 18px">
        <span>后端连接失败：{{ store.bootError }}</span>
      </div>

      <div v-if="store.health && !store.health.ffmpeg" class="alert alert-warn" style="margin-bottom: 18px">
        <span>
          未找到 ffmpeg，抽帧功能不可用。把 <code class="mono">ffmpeg.exe</code> 放进
          <code class="mono">backend/runtime/</code>，或在
          <code class="mono">backend/.env</code> 里设置 <code class="mono">FFMPEG_PATH</code>。
        </span>
      </div>

      <div
        v-if="store.health && !store.health.vlm_configured"
        class="alert alert-warn"
        style="margin-bottom: 18px"
      >
        <span>
          未配置多模态模型。在 <code class="mono">backend/.env</code> 里填写
          <code class="mono">VLM_API_KEY</code> / <code class="mono">VLM_BASE_URL</code> /
          <code class="mono">VLM_MODEL</code> 后重启服务。
        </span>
      </div>

      <RouterView />
    </main>
  </div>
</template>
