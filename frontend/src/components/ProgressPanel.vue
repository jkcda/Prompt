<script setup lang="ts">
import { computed, ref } from 'vue'

import { useAnalyzeStore } from '@/stores/analyze'

const store = useAnalyzeStore()
const showLog = ref(false)

/** 后端阶段顺序，用于渲染步骤条。 */
const STAGES = [
  { key: 'fetch', label: '抓取视频' },
  { key: 'probe', label: '探测媒体' },
  { key: 'scenes', label: '镜头切分' },
  { key: 'audio', label: '音频分析' },
  { key: 'plan', label: '帧预算分配' },
  { key: 'frames', label: '抽取关键帧' },
  { key: 'observe', label: '逐块视觉分析' },
  { key: 'compose', label: '合成提示词' },
]

const currentStage = computed(() => store.job?.progress.stage ?? '')

function stageState(key: string): 'done' | 'active' | 'todo' {
  const order = STAGES.map((s) => s.key)
  const cur = order.indexOf(currentStage.value)
  const mine = order.indexOf(key)
  if (cur === -1) return 'todo'
  if (store.job?.state === 'succeeded') return 'done'
  if (mine < cur) return 'done'
  if (mine === cur) return 'active'
  return 'todo'
}

const visibleStages = computed(() =>
  // 上传任务没有「抓取」阶段，隐藏掉避免误导
  store.job?.source === 'upload' ? STAGES.filter((s) => s.key !== 'fetch') : STAGES,
)

const stateClass = computed(() => {
  const s = store.job?.state
  if (s === 'succeeded') return 'done'
  if (s === 'failed') return 'failed'
  return ''
})

const logs = computed(() =>
  store.events
    .filter((e) => e.type === 'progress' || e.type === 'chunk' || e.type === 'fetched')
    .slice(-60),
)

function eventText(e: (typeof store.events)[number]): string {
  if (e.type === 'chunk') return `分块 ${(e.index ?? 0) + 1}/${e.total} 分析完成，得到 ${e.shots} 个镜头`
  if (e.type === 'fetched') return `已抓取：${e.title ?? ''}`
  return e.message || e.stage_label || ''
}
</script>

<template>
  <section v-if="store.job || store.error" class="panel">
    <div class="panel-head">
      <div class="panel-title">
        <span class="dot" :class="{ pulsing: store.running }" />
        反推进度
      </div>
      <div class="row" style="gap: 8px">
        <span v-if="store.running" class="badge info">
          <span class="spinner" /> 进行中 {{ store.elapsed }}s
        </span>
        <span v-else-if="store.job?.state === 'succeeded'" class="badge ok">完成</span>
        <span v-else-if="store.job?.state === 'failed'" class="badge err">失败</span>
        <span v-else-if="store.job?.state === 'cancelled'" class="badge warn">已取消</span>
        <button v-if="store.running" class="btn btn-sm btn-danger" @click="store.cancel()">
          取消
        </button>
      </div>
    </div>

    <div class="panel-body">
      <div v-if="store.error" class="alert alert-err" style="margin-bottom: 16px">
        <span style="white-space: pre-wrap">{{ store.error }}</span>
      </div>

      <div v-if="store.job?.error" class="alert alert-err" style="margin-bottom: 16px">
        <span style="white-space: pre-wrap">{{ store.job.error }}</span>
      </div>

      <template v-if="store.job">
        <div class="row" style="justify-content: space-between; margin-bottom: 7px">
          <span class="stage-hint">{{ store.stageHint }}</span>
          <span class="mono faint">{{ store.progress }}%</span>
        </div>
        <div class="progress-track">
          <div
            :class="['progress-fill', stateClass]"
            :style="{ width: Math.max(2, store.progress) + '%' }"
          />
        </div>

        <!-- 步骤条 -->
        <div class="stages">
          <div
            v-for="s in visibleStages"
            :key="s.key"
            :class="['stage', stageState(s.key)]"
          >
            <span class="stage-dot" />
            <span class="stage-name">{{ s.label }}</span>
          </div>
        </div>

        <button class="btn btn-ghost btn-sm" style="margin-top: 14px" @click="showLog = !showLog">
          {{ showLog ? '收起' : '展开' }}运行日志（{{ logs.length }}）
        </button>

        <div v-if="showLog" class="log">
          <div v-for="(e, i) in logs" :key="i" class="log-line">
            <span class="faint mono">{{ new Date(e.ts * 1000).toLocaleTimeString('zh-CN') }}</span>
            <span>{{ eventText(e) }}</span>
          </div>
          <div v-if="!logs.length" class="faint">暂无日志</div>
        </div>
      </template>
    </div>
  </section>
</template>

<style scoped>
.stage-hint {
  font-size: 13px;
  font-weight: 500;
}

.stages {
  display: flex;
  flex-wrap: wrap;
  gap: 6px 16px;
  margin-top: 16px;
}

.stage {
  display: flex;
  align-items: center;
  gap: 7px;
  font-size: 12px;
  color: var(--text-faint);
}

.stage-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--border);
  flex-shrink: 0;
}

.stage.done {
  color: var(--text-dim);
}
.stage.done .stage-dot {
  background: var(--ok);
}

.stage.active {
  color: var(--accent-hover);
  font-weight: 500;
}
.stage.active .stage-dot {
  background: var(--accent);
  box-shadow: 0 0 0 3px var(--accent-soft);
}

.log {
  margin-top: 10px;
  padding: 11px 13px;
  max-height: 200px;
  overflow-y: auto;
  border: 1px solid var(--border-soft);
  border-radius: var(--radius);
  background: var(--bg-soft);
  font-size: 12px;
}

.log-line {
  display: flex;
  gap: 10px;
  padding: 2px 0;
  line-height: 1.6;
}
</style>
