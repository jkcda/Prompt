<script setup lang="ts">
import { computed } from 'vue'
import { useRouter } from 'vue-router'

import { useAnalyzeStore } from '@/stores/analyze'
import type { JobSummary } from '@/types'

const store = useAnalyzeStore()
const router = useRouter()

const items = computed(() => store.history)

const STATE_META: Record<string, { cls: string; text: string }> = {
  succeeded: { cls: 'ok', text: '完成' },
  running: { cls: 'info', text: '进行中' },
  pending: { cls: 'info', text: '排队中' },
  failed: { cls: 'err', text: '失败' },
  cancelled: { cls: 'warn', text: '已取消' },
}

function meta(state: string) {
  return STATE_META[state] ?? { cls: '', text: state }
}

function when(ts: number): string {
  const diff = Date.now() / 1000 - ts
  if (diff < 60) return '刚刚'
  if (diff < 3600) return `${Math.floor(diff / 60)} 分钟前`
  if (diff < 86400) return `${Math.floor(diff / 3600)} 小时前`
  return new Date(ts * 1000).toLocaleDateString('zh-CN')
}

function open(job: JobSummary) {
  if (job.state === 'running' || job.state === 'pending') {
    void store.loadJob(job.id)
    return
  }
  void router.push({ name: 'result', params: { id: job.id } })
}

async function remove(job: JobSummary, e: Event) {
  e.stopPropagation()
  await store.remove(job.id)
}
</script>

<template>
  <section class="panel">
    <div class="panel-head">
      <div class="panel-title"><span class="dot" />历史记录</div>
      <button class="btn btn-ghost btn-sm" @click="store.loadHistory()">刷新</button>
    </div>

    <div v-if="!items.length" class="empty">
      {{ store.historyLoading ? '加载中…' : '还没有反推记录' }}
    </div>

    <div v-else class="history-list">
      <div
        v-for="job in items"
        :key="job.id"
        class="history-item"
        :class="{ active: job.id === store.jobId }"
        @click="open(job)"
      >
        <div class="hi-top">
          <span :class="['badge', meta(job.state).cls]">{{ meta(job.state).text }}</span>
          <span class="faint mono hi-time">{{ when(job.created_at) }}</span>
          <button class="hi-del" title="删除记录" @click="remove(job, $event)">×</button>
        </div>

        <div class="hi-title">{{ job.title || '未命名' }}</div>

        <div v-if="job.prompt_preview" class="hi-preview">
          {{ job.prompt_preview }}
        </div>
        <div v-else-if="job.error" class="hi-preview err">{{ job.error }}</div>

        <div class="hi-meta faint">
          <span v-if="job.format" class="mono">{{ store.labelOfFormat(job.format) }}</span>
          <span v-if="job.frames_used">{{ job.frames_used }} 帧</span>
          <span v-if="job.elapsed_sec">{{ job.elapsed_sec }}s</span>
          <span v-if="job.source && job.source !== 'upload'">{{ job.source }}</span>
        </div>
      </div>
    </div>
  </section>
</template>

<style scoped>
.history-list {
  max-height: 620px;
  overflow-y: auto;
}

.history-item {
  padding: 11px 16px;
  border-bottom: 1px solid var(--border-soft);
  cursor: pointer;
  transition: background 0.13s;
}

.history-item:last-child {
  border-bottom: none;
}

.history-item:hover {
  background: var(--panel-2);
}

.history-item.active {
  background: var(--accent-soft);
  box-shadow: inset 2px 0 0 var(--accent);
}

.hi-top {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 5px;
}

.hi-time {
  font-size: 11px;
  margin-left: auto;
}

.hi-del {
  width: 18px;
  height: 18px;
  display: grid;
  place-items: center;
  border: none;
  border-radius: 4px;
  background: transparent;
  color: var(--text-faint);
  font-size: 15px;
  line-height: 1;
  cursor: pointer;
  opacity: 0;
  transition: opacity 0.13s, color 0.13s;
}

.history-item:hover .hi-del {
  opacity: 1;
}

.hi-del:hover {
  color: var(--err);
  background: rgba(255, 107, 107, 0.12);
}

.hi-title {
  font-size: 13px;
  font-weight: 500;
  line-height: 1.45;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.hi-preview {
  margin-top: 4px;
  font-size: 11.5px;
  line-height: 1.55;
  color: var(--text-faint);
  display: -webkit-box;
  -webkit-line-clamp: 2;
  line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.hi-preview.err {
  color: #ff9b9b;
}

.hi-meta {
  display: flex;
  gap: 12px;
  margin-top: 6px;
  font-size: 11px;
}
</style>
