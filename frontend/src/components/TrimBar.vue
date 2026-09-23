<script setup lang="ts">
/**
 * 片段选择：预览视频 + 框选区间。
 *
 * 为什么要有这个：抓取/上传的视频经常前后带无关内容（片头、广告、其他片段），
 * 整片丢给模型会把无关内容也写进提示词。让用户先看再框选，产出才对得上他要的那段。
 *
 * 两种操作方式并存：
 *   - 拖时间轴上的两个手柄（粗调，直观）
 *   - 播放到某处点「设为起点 / 终点」（精调，能卡到具体动作）
 */
import { computed, ref, watch } from 'vue'

const props = defineProps<{
  src: string
  /** 整片时长（秒）。0 表示还没探测出来。 */
  duration: number
  /** 当前选区。null 表示整片。 */
  modelValue: { start: number; end: number } | null
}>()

const emit = defineEmits<{
  (e: 'update:modelValue', v: { start: number; end: number } | null): void
}>()

const video = ref<HTMLVideoElement | null>(null)
const track = ref<HTMLDivElement | null>(null)
const playhead = ref(0)
const dragging = ref<'start' | 'end' | null>(null)

/**
 * 从 `<video>` 元素读到的时长。
 *
 * ⚠️ 必须存在 ref 里，不能直接在 computed 里读 `video.value.duration` ——
 * **`<video>` 的 duration 不是响应式的**，Vue 追踪不到它。
 * 踩过：上传的视频没有探测结果（probe 只用于链接模式），`props.duration` 是 0，
 * 而 computed 又读不到元素上的 duration，于是 `total` 永远是 0，
 * 拖动时被 `total <= 0` 直接挡掉 —— 表现就是「选不了时间段」。
 */
const videoDuration = ref(0)

/** 整片时长优先用探测值，没有就用 video 元素报的。 */
const total = computed(() => (props.duration > 0 ? props.duration : videoDuration.value))

/**
 * 交互时刻**直接**读一次时长，作为兜底。
 *
 * 即使某个环节的响应式没跟上（元数据刚加载完、computed 还没重算），
 * 拖动也不会被误挡。
 */
function currentTotal(): number {
  if (props.duration > 0) return props.duration
  if (videoDuration.value > 0) return videoDuration.value
  const v = video.value
  return v && isFinite(v.duration) && v.duration > 0 ? v.duration : 0
}

const start = computed(() => props.modelValue?.start ?? 0)
const end = computed(() => props.modelValue?.end ?? total.value)
const selected = computed(() => Math.max(0, end.value - start.value))

const pct = (t: number) => (total.value > 0 ? Math.min(100, Math.max(0, (t / total.value) * 100)) : 0)

const fmt = (t: number) => {
  const s = Math.max(0, t)
  const m = Math.floor(s / 60)
  const rest = s - m * 60
  return m > 0
    ? `${m}:${rest.toFixed(2).padStart(5, '0')}`
    : `${rest.toFixed(2)}s`
}

/** 选区至少 0.5 秒 —— 后端也校验，这里先拦住避免白跑一次请求。 */
const MIN_LEN = 0.5

function setRange(a: number, b: number) {
  const span = currentTotal()
  const lo = Math.max(0, Math.min(a, b))
  const hi = Math.min(span > 0 ? span : Math.max(a, b), Math.max(a, b))
  if (hi - lo < MIN_LEN) return
  emit('update:modelValue', { start: lo, end: hi })
}

function onTrackPointerDown(e: PointerEvent) {
  const span = currentTotal()
  if (!track.value || span <= 0) return
  const rect = track.value.getBoundingClientRect()
  const t = ((e.clientX - rect.left) / rect.width) * span
  // 点哪边近就动哪个手柄
  const toStart = Math.abs(t - start.value)
  const toEnd = Math.abs(t - end.value)
  const which = toStart <= toEnd ? 'start' : 'end'
  dragging.value = which
  moveTo(which, t)
  ;(e.target as HTMLElement).setPointerCapture?.(e.pointerId)
}

function onTrackPointerMove(e: PointerEvent) {
  const span = currentTotal()
  if (!dragging.value || !track.value || span <= 0) return
  const rect = track.value.getBoundingClientRect()
  moveTo(dragging.value, ((e.clientX - rect.left) / rect.width) * span)
}

function moveTo(which: 'start' | 'end', t: number) {
  const span = currentTotal()
  const clamped = span > 0 ? Math.min(span, Math.max(0, t)) : Math.max(0, t)
  if (which === 'start') {
    setRange(clamped, end.value)
  } else {
    setRange(start.value, clamped)
  }
}

function onTrackPointerUp() {
  dragging.value = null
}

function usePlayheadAs(which: 'start' | 'end') {
  if (which === 'start') setRange(playhead.value, end.value)
  else setRange(start.value, playhead.value)
}

function selectAll() {
  emit('update:modelValue', null)
}

/** 预览选区：从头播到尾部就停，方便确认框对了。 */
function playSelection() {
  const v = video.value
  if (!v || currentTotal() <= 0) return
  v.currentTime = start.value
  void v.play()
}

function onTimeUpdate() {
  const v = video.value
  if (!v) return
  playhead.value = v.currentTime
  // 播过选区尾部就停下，不然会一直播到片尾
  if (props.modelValue && v.currentTime >= end.value && !v.paused) v.pause()
}

function onLoadedMetadata() {
  const v = video.value
  if (!v) return
  // 存进 ref —— `<video>` 的 duration 不是响应式的，不存的话 total 永远是 0
  videoDuration.value = isFinite(v.duration) && v.duration > 0 ? v.duration : 0

  if (props.modelValue) return
  // 没框选时，如果视频比上限长，默认帮用户框出前段
  const limit = props.duration > 0 ? props.duration : 0
  if (limit > 0 && videoDuration.value > limit + 0.05) {
    emit('update:modelValue', { start: 0, end: limit })
  }
}

// 换源时重置播放头
watch(() => props.src, () => {
  playhead.value = 0
})
</script>

<template>
  <div class="trim">
    <div class="trim-video">
      <video
        ref="video"
        :src="src"
        controls
        preload="metadata"
        @timeupdate="onTimeUpdate"
        @loadedmetadata="onLoadedMetadata"
      />
    </div>

    <div
      ref="track"
      class="trim-track"
      :class="{ dragging: dragging !== null }"
      @pointerdown="onTrackPointerDown"
      @pointermove="onTrackPointerMove"
      @pointerup="onTrackPointerUp"
      @pointercancel="onTrackPointerUp"
    >
      <div class="trim-fill" :style="{ left: pct(start) + '%', width: pct(end - start) + '%' }" />
      <div class="trim-head" :style="{ left: pct(playhead) + '%' }" />
      <div class="trim-handle" :style="{ left: pct(start) + '%' }" title="起点" />
      <div class="trim-handle" :style="{ left: pct(end) + '%' }" title="终点" />
    </div>

    <div class="trim-row">
      <span class="faint mono">{{ fmt(start) }}</span>
      <span class="trim-sep">→</span>
      <span class="faint mono">{{ fmt(end) }}</span>
      <span class="badge" :class="{ warn: selected < 1 }">
        选中 {{ fmt(selected) }}
      </span>
      <span v-if="modelValue" class="faint">（未选则是整片）</span>
      <span v-else class="faint">整片 {{ fmt(total) }}</span>
    </div>

    <div class="trim-actions">
      <button class="btn tiny" @click="usePlayheadAs('start')">设为起点</button>
      <button class="btn tiny" @click="usePlayheadAs('end')">设为终点</button>
      <button class="btn tiny" @click="playSelection">预览选区</button>
      <button class="btn tiny" :disabled="!modelValue" @click="selectAll">用整片</button>
    </div>
  </div>
</template>

<style scoped>
.trim {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.trim-video video {
  width: 100%;
  max-height: 260px;
  border-radius: 8px;
  background: #000;
  display: block;
}

.trim-track {
  position: relative;
  height: 34px;
  border-radius: 6px;
  background: var(--panel-2);
  border: 1px solid var(--border);
  cursor: pointer;
  touch-action: none;
  user-select: none;
}

.trim-track.dragging {
  cursor: ew-resize;
}

.trim-fill {
  position: absolute;
  top: 0;
  bottom: 0;
  background: var(--accent-soft);
  border-left: 2px solid var(--accent);
  border-right: 2px solid var(--accent);
  pointer-events: none;
}

.trim-head {
  position: absolute;
  top: -3px;
  bottom: -3px;
  width: 2px;
  margin-left: -1px;
  background: var(--text);
  opacity: 0.85;
  pointer-events: none;
}

.trim-handle {
  position: absolute;
  top: 50%;
  width: 14px;
  height: 22px;
  margin: -11px 0 0 -7px;
  border-radius: 4px;
  background: var(--accent);
  border: 1px solid var(--accent-hover);
  pointer-events: none;
}

.trim-row {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
}

.trim-sep {
  color: var(--text-faint);
}

.trim-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
</style>
