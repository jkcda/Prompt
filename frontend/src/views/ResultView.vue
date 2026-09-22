<script setup lang="ts">
import { onMounted, watch } from 'vue'
import { RouterLink, useRouter } from 'vue-router'

import PromptPanel from '@/components/PromptPanel.vue'
import ProgressPanel from '@/components/ProgressPanel.vue'
import VideoPanel from '@/components/VideoPanel.vue'
import { useAnalyzeStore } from '@/stores/analyze'

const props = defineProps<{ id: string }>()
const store = useAnalyzeStore()
const router = useRouter()

onMounted(() => {
  if (store.jobId !== props.id) void store.loadJob(props.id)
})

// 同一个组件复用时（历史列表里连续点两条）也要重新加载
watch(
  () => props.id,
  (id) => {
    if (store.jobId !== id) void store.loadJob(id)
  },
)

function back() {
  void router.push({ name: 'home' })
}
</script>

<template>
  <div>
    <div class="toolbar">
      <button class="btn btn-sm" @click="back">← 返回</button>
      <div v-if="store.job" class="row" style="gap: 10px; min-width: 0">
        <span class="title">{{ store.job.title || '未命名' }}</span>
        <span class="faint mono">#{{ store.job.id }}</span>
      </div>
      <RouterLink to="/" class="btn btn-primary btn-sm" style="margin-left: auto">
        新建反推
      </RouterLink>
    </div>

    <div v-if="store.error" class="alert alert-err" style="margin-bottom: 18px">
      {{ store.error }}
    </div>

    <div class="split">
      <div class="col">
        <VideoPanel />
      </div>
      <div class="col">
        <ProgressPanel />
        <PromptPanel />
      </div>
    </div>
  </div>
</template>

<style scoped>
.toolbar {
  display: flex;
  align-items: center;
  gap: 14px;
  margin-bottom: 18px;
  min-width: 0;
}

.title {
  font-size: 15px;
  font-weight: 600;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.col {
  display: flex;
  flex-direction: column;
  gap: 20px;
  min-width: 0;
}
</style>
