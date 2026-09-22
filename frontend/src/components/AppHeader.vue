<script setup lang="ts">
import { computed } from 'vue'
import { RouterLink } from 'vue-router'

import { useAnalyzeStore } from '@/stores/analyze'

defineProps<{ isHome: boolean }>()

const store = useAnalyzeStore()

const vlmBadge = computed(() => {
  const h = store.health
  if (!h) return { cls: '', text: '连接中…' }
  return h.vlm_configured
    ? { cls: 'ok', text: h.vlm_model.split('/').pop() ?? '模型就绪' }
    : { cls: 'err', text: '模型未配置' }
})

const asrBadge = computed(() => {
  const h = store.health
  if (!h) return null
  return h.asr_configured ? { cls: 'ok', text: '语音转写已启用' } : null
})
</script>

<template>
  <header class="header">
    <div class="header-inner">
      <RouterLink to="/" class="brand">
        <span class="brand-mark">反</span>
        <span class="brand-text">
          <strong>视频提示词反推</strong>
          <small>ffmpeg 镜头分割 · 两阶段多模态反推</small>
        </span>
      </RouterLink>

      <div class="header-right">
        <span v-if="store.health?.ytdlp" class="badge info">链接抓取可用</span>
        <span v-if="asrBadge" :class="['badge', asrBadge.cls]">{{ asrBadge.text }}</span>
        <span :class="['badge', vlmBadge.cls]">{{ vlmBadge.text }}</span>

        <RouterLink v-if="!isHome" to="/" class="btn btn-sm">新建反推</RouterLink>
      </div>
    </div>
  </header>
</template>

<style scoped>
.header {
  position: sticky;
  top: 0;
  z-index: 20;
  background: rgba(11, 13, 18, 0.86);
  backdrop-filter: blur(14px);
  border-bottom: 1px solid var(--border-soft);
}

.header-inner {
  max-width: 1560px;
  margin: 0 auto;
  padding: 11px 24px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
}

.brand {
  display: flex;
  align-items: center;
  gap: 11px;
  color: var(--text);
}

.brand:hover {
  color: var(--text);
}

.brand-mark {
  width: 32px;
  height: 32px;
  display: grid;
  place-items: center;
  border-radius: 9px;
  background: linear-gradient(135deg, var(--accent), #9d7bff);
  color: #0a0d16;
  font-weight: 700;
  font-size: 15px;
  flex-shrink: 0;
}

.brand-text {
  display: flex;
  flex-direction: column;
  line-height: 1.25;
}

.brand-text strong {
  font-size: 14.5px;
  letter-spacing: 0.2px;
}

.brand-text small {
  font-size: 11px;
  color: var(--text-faint);
}

.header-right {
  display: flex;
  align-items: center;
  gap: 8px;
}

@media (max-width: 720px) {
  .brand-text small,
  .header-right .badge {
    display: none;
  }
}
</style>
