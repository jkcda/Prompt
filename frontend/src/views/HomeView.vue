<script setup lang="ts">
import HistoryList from '@/components/HistoryList.vue'
import PromptPanel from '@/components/PromptPanel.vue'
import ProgressPanel from '@/components/ProgressPanel.vue'
import SourcePicker from '@/components/SourcePicker.vue'
import VideoPanel from '@/components/VideoPanel.vue'
</script>

<template>
  <div class="home">
    <!-- 左栏：选源 + 选项 + 历史 -->
    <div class="col">
      <SourcePicker />
      <HistoryList />
    </div>

    <!-- 右栏：进度 + 结果 -->
    <div class="col">
      <ProgressPanel />
      <VideoPanel />
      <PromptPanel />
    </div>
  </div>
</template>

<style scoped>
.home {
  display: grid;
  grid-template-columns: minmax(340px, 400px) minmax(0, 1fr);
  gap: 20px;
  align-items: start;
}

.col {
  display: flex;
  flex-direction: column;
  gap: 20px;
  min-width: 0;
}

@media (max-width: 1180px) {
  .home {
    grid-template-columns: 1fr;
  }
}
</style>
