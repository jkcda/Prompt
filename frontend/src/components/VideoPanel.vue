<script setup lang="ts">
import { computed, ref, watch } from 'vue'

import { useAnalyzeStore } from '@/stores/analyze'

const store = useAnalyzeStore()
const videoEl = ref<HTMLVideoElement | null>(null)
const currentTime = ref(0)
const showFrames = ref(true)

const media = computed(() => store.media)
const shots = computed(() => store.shots)
const frames = computed(() => store.frameUrls)

const metaLine = computed(() => {
  const m = media.value
  if (!m) return ''
  const parts = [`${m.width}×${m.height}`, `${m.fps.toFixed(2)}fps`]
  if (m.video_codec) parts.push(m.video_codec)
  if (m.has_audio) parts.push(`音频 ${m.audio_codec || '有'}`)
  else parts.push('无音轨')
  return parts.join(' · ')
})

const durationText = computed(() => {
  const d = media.value?.duration ?? store.uploadResult?.duration ?? 0
  return d ? formatTime(d) : '—'
})

function formatTime(sec: number): string {
  const s = Math.max(0, sec)
  const m = Math.floor(s / 60)
  const r = s - m * 60
  return `${String(m).padStart(2, '0')}:${r.toFixed(2).padStart(5, '0')}`
}

function onTimeUpdate() {
  if (videoEl.value) currentTime.value = videoEl.value.currentTime
}

function seek(t: number) {
  const v = videoEl.value
  if (!v) return
  v.currentTime = Math.max(0, t)
  void v.play().catch(() => {
    /* 自动播放可能被拦，忽略 */
  })
}

/** 抽帧图在时间轴上的位置（0~100%），用于帧带里的小刻度。 */
function framePercent(url: string): number {
  const t = timeOfFrame(url)
  const total = media.value?.duration ?? 0
  if (!total) return 0
  return Math.min(100, (t / total) * 100)
}

function timeOfFrame(url: string): number {
  const m = url.match(/t(\d{9})\.jpg$/)
  if (!m) return 0
  return Number(m[1]) / 1000
}

watch(
  () => store.videoUrl,
  () => {
    currentTime.value = 0
  },
)
</script>

<template>
  <section class="panel">
    <div class="panel-head">
      <div class="panel-title"><span class="dot" />源视频</div>
      <div class="row" style="gap: 8px">
        <span v-if="media" class="badge">{{ durationText }}</span>
        <span v-if="shots.length" class="badge info">{{ shots.length }} 个镜头</span>
      </div>
    </div>

    <div class="video-wrap">
      <video
        v-if="store.videoUrl"
        ref="videoEl"
        :src="store.videoUrl"
        controls
        preload="metadata"
        playsinline
        @timeupdate="onTimeUpdate"
      />
      <div v-else class="empty">上传视频或解析链接后，这里会显示源视频</div>
    </div>

    <div v-if="media || store.uploadResult" class="meta-bar">
      <div class="meta-line">
        <span class="faint">时长</span>
        <span class="mono">{{ durationText }}</span>
      </div>
      <div class="meta-line">
        <span class="faint">规格</span>
        <span class="mono">{{ metaLine }}</span>
      </div>
      <div v-if="store.job?.title" class="meta-line">
        <span class="faint">标题</span>
        <span class="ellipsis">{{ store.job.title }}</span>
      </div>
    </div>

    <!-- ------------------------------------------------ 抽帧带 -->
    <div v-if="frames.length" class="frames-block">
      <div class="frames-head">
        <span class="faint">
          实际送入模型的 {{ frames.length }} 张帧（点击可跳转）
        </span>
        <button class="btn btn-ghost btn-sm" @click="showFrames = !showFrames">
          {{ showFrames ? '收起' : '展开' }}
        </button>
      </div>

      <div v-if="showFrames" class="frames-scroll">
        <div
          v-for="(url, i) in frames"
          :key="url"
          :class="['frame-cell', { active: Math.abs(timeOfFrame(url) - currentTime) < 0.5 }]"
          :title="`${formatTime(timeOfFrame(url))}（第 ${i + 1} 张）`"
          @click="seek(timeOfFrame(url))"
        >
          <img :src="url" loading="lazy" alt="" />
          <span class="frame-time mono">{{ timeOfFrame(url).toFixed(1) }}s</span>
        </div>
      </div>

      <!-- 时间轴刻度：把抽帧分布和镜头边界画出来，方便判断抽帧是否合理 -->
      <div v-if="showFrames && media?.duration" class="timeline">
        <div
          v-for="(f, i) in frames"
          :key="'tick' + i"
          class="tick"
          :style="{ left: framePercent(f) + '%' }"
        />
        <div
          v-for="s in shots.slice(1)"
          :key="'cut' + s.index"
          class="cut"
          :style="{ left: (s.start / (media?.duration || 1)) * 100 + '%' }"
          :title="`切点 ${formatTime(s.start)}`"
        />
      </div>
    </div>
  </section>
</template>

<style scoped>
.video-wrap {
  background: #000;
  aspect-ratio: 16 / 9;
  display: flex;
  align-items: center;
  justify-content: center;
}

.video-wrap video {
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
}

.video-wrap .empty {
  color: var(--text-faint);
}

.meta-bar {
  display: flex;
  flex-wrap: wrap;
  gap: 6px 22px;
  padding: 11px 18px;
  border-top: 1px solid var(--border-soft);
  background: var(--bg-soft);
}

.meta-line {
  display: flex;
  align-items: baseline;
  gap: 8px;
  font-size: 12.5px;
  min-width: 0;
}

.ellipsis {
  max-width: 420px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.frames-block {
  padding: 14px 18px 18px;
  border-top: 1px solid var(--border-soft);
}

.frames-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 10px;
  font-size: 12px;
}

.frames-scroll {
  display: flex;
  gap: 8px;
  overflow-x: auto;
  padding-bottom: 8px;
}

.frame-cell {
  position: relative;
  flex: 0 0 auto;
  width: 118px;
  border-radius: var(--radius-sm);
  overflow: hidden;
  border: 1px solid var(--border);
  cursor: pointer;
  transition: border-color 0.14s, transform 0.1s;
  background: #000;
}

.frame-cell:hover {
  border-color: var(--accent);
  transform: translateY(-2px);
}

.frame-cell.active {
  border-color: var(--accent);
  box-shadow: 0 0 0 2px var(--accent-soft);
}

.frame-cell img {
  display: block;
  width: 100%;
  height: 66px;
  object-fit: cover;
}

.frame-time {
  position: absolute;
  left: 4px;
  bottom: 3px;
  padding: 0 5px;
  border-radius: 3px;
  font-size: 10px;
  color: #fff;
  background: rgba(0, 0, 0, 0.68);
}

.timeline {
  position: relative;
  height: 18px;
  margin-top: 10px;
  border-radius: 4px;
  background: var(--bg-soft);
  border: 1px solid var(--border-soft);
}

.tick {
  position: absolute;
  top: 3px;
  width: 2px;
  height: 10px;
  margin-left: -1px;
  border-radius: 1px;
  background: var(--accent);
  opacity: 0.85;
}

.cut {
  position: absolute;
  top: 0;
  width: 1px;
  height: 100%;
  background: var(--warn);
  opacity: 0.7;
}
</style>
