<script setup lang="ts">
import { computed, ref } from 'vue'

import { errorMessage, savePrompt } from '@/api'
import { useAnalyzeStore } from '@/stores/analyze'

const store = useAnalyzeStore()

const tab = ref<'prompt' | 'shots' | 'subjects' | 'audio'>('prompt')
const copied = ref(false)
const saving = ref(false)
const saveMsg = ref('')
const saveError = ref('')

const prompt = computed(() => store.prompt)

/** 结果里记的 format 优先（历史任务可能不是当前选的模式）。 */
const activeFormat = computed(() => (store.job?.options.format ?? store.options.format) as string)

/** 「H3 模式 · Ref2VA 六段式（带参考素材）」这样的完整称呼。 */
const formatLabel = computed(() => {
  for (const m of store.modes) {
    const v = m.variants.find((x) => x.value === activeFormat.value)
    if (v) return v.label === m.label ? m.label : `${m.label} · ${v.label}`
  }
  return activeFormat.value
})

/** 单变体模式下前缀冗余，徽标只显示模式名。 */
const modeBadge = computed(() => {
  for (const m of store.modes) {
    if (m.variants.some((x) => x.value === activeFormat.value)) {
      return m.variants.length > 1 ? formatLabel.value : m.label
    }
  }
  return activeFormat.value
})

const wordCount = computed(() => (prompt.value ? prompt.value.split(/\s+/).filter(Boolean).length : 0))
const charCount = computed(() => prompt.value.length)

/** 各格式的段落切分，用于侧栏做锚点导航。 */
const sections = computed(() => {
  if (!prompt.value) return []
  const known = [
    'subject_definitions', 'summary', 'retention_analysis',
    'detailed_description', 'overall_soundscape', 'non_diegetic_music',
    'integrated_multimodal_description',
  ]
  const lines = prompt.value.split('\n')
  const out: Array<{ label: string; line: number }> = []
  lines.forEach((l, i) => {
    const m = l.match(/^([a-z_]+):/i)
    if (m && known.includes(m[1].toLowerCase())) out.push({ label: m[1], line: i })
  })
  return out
})

async function copy() {
  if (!prompt.value) return
  try {
    await navigator.clipboard.writeText(prompt.value)
    copied.value = true
    window.setTimeout(() => (copied.value = false), 1600)
  } catch {
    saveError.value = '复制失败，请手动选中文本'
  }
}

function download() {
  if (!prompt.value) return
  const blob = new Blob([prompt.value], { type: 'text/plain;charset=utf-8' })
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = `prompt-${store.jobId || 'draft'}.txt`
  a.click()
  URL.revokeObjectURL(a.href)
}

async function save() {
  if (!prompt.value || !store.jobId) return
  saving.value = true
  saveMsg.value = ''
  saveError.value = ''
  try {
    await savePrompt({
      job_id: store.jobId,
      content: prompt.value,
      title: store.job?.title ?? '',
      format: (store.job?.options.format ?? 'h3') as string,
    })
    saveMsg.value = '已收藏到提示词库'
    window.setTimeout(() => (saveMsg.value = ''), 2000)
  } catch (e) {
    saveError.value = errorMessage(e)
  } finally {
    saving.value = false
  }
}

function scrollToSection(line: number) {
  const box = document.querySelector<HTMLElement>('.prompt-box')
  if (!box) return
  const lineHeight = parseFloat(getComputedStyle(box).lineHeight) || 24
  box.scrollTo({ top: line * lineHeight, behavior: 'smooth' })
}

function fmtTime(sec: number): string {
  const m = Math.floor(sec / 60)
  const s = sec - m * 60
  return `${String(m).padStart(2, '0')}:${s.toFixed(1).padStart(4, '0')}`
}
</script>

<template>
  <section class="panel">
    <div class="panel-head">
      <div class="panel-title"><span class="dot" />反推提示词</div>
      <div class="row" style="gap: 8px">
        <span v-if="prompt" class="badge">{{ modeBadge }}</span>
        <span v-if="prompt" class="badge">{{ charCount }} 字</span>
      </div>
    </div>

    <div class="panel-body">
      <!-- ------------------------------------------------ 未出结果 -->
      <div v-if="!prompt" class="empty">
        <template v-if="store.running">
          <span class="spinner" style="display: inline-block; margin-bottom: 10px" />
          <div>{{ store.stageHint || '正在反推…' }}</div>
        </template>
        <template v-else>
          反推完成后，提示词会显示在这里
        </template>
      </div>

      <!-- ------------------------------------------------ 出结果 -->
      <template v-else>
        <div class="tabs" style="margin-bottom: 14px">
          <button :class="['tab', { active: tab === 'prompt' }]" @click="tab = 'prompt'">
            提示词
          </button>
          <button
            :class="['tab', { active: tab === 'shots' }]"
            @click="tab = 'shots'"
          >
            镜头观察 {{ store.observations.length }}
          </button>
          <button
            v-if="store.subjects.length"
            :class="['tab', { active: tab === 'subjects' }]"
            @click="tab = 'subjects'"
          >
            主体 {{ store.subjects.length }}
          </button>
          <button :class="['tab', { active: tab === 'audio' }]" @click="tab = 'audio'">
            音频
          </button>
        </div>

        <!-- 提示词 -->
        <template v-if="tab === 'prompt'">
          <div class="row row-wrap" style="gap: 8px; margin-bottom: 12px">
            <button class="btn btn-primary btn-sm" @click="copy">
              {{ copied ? '已复制' : '复制全文' }}
            </button>
            <button class="btn btn-sm" @click="download">下载 .txt</button>
            <button class="btn btn-sm" :disabled="saving || !store.jobId" @click="save">
              <span v-if="saving" class="spinner" />
              {{ saving ? '保存中' : '收藏' }}
            </button>
            <span v-if="saveMsg" class="badge ok">{{ saveMsg }}</span>
            <span v-if="saveError" class="badge err">{{ saveError }}</span>
          </div>

          <div v-if="sections.length" class="anchors">
            <button
              v-for="s in sections"
              :key="s.label"
              class="anchor"
              @click="scrollToSection(s.line)"
            >
              {{ s.label }}
            </button>
          </div>

          <div class="prompt-box">{{ prompt }}</div>

          <div class="stats-row">
            <span class="faint">约 {{ wordCount }} 词</span>
            <span v-if="store.stats.frames" class="faint">
              {{ store.stats.frames }} 帧
            </span>
            <span v-if="store.stats.shots" class="faint">
              {{ store.stats.shots }} 镜头
            </span>
            <span v-if="store.stats.elapsed_sec" class="faint">
              耗时 {{ store.stats.elapsed_sec }}s
            </span>
          </div>
        </template>

        <!-- 镜头观察 -->
        <template v-else-if="tab === 'shots'">
          <div class="shot-list">
            <div v-for="(o, i) in store.observations" :key="i" class="shot-card">
              <div class="shot-head">
                <span class="shot-no">镜头 {{ o.shot || i + 1 }}</span>
                <span v-if="o.timecode" class="mono faint">{{ o.timecode }}</span>
                <span v-if="o.shot_size" class="badge">{{ o.shot_size }}</span>
                <span v-if="o.confidence" class="badge" :class="o.confidence < 0.6 ? 'warn' : ''">
                  置信 {{ (o.confidence * 100).toFixed(0) }}%
                </span>
              </div>
              <dl class="shot-fields">
                <template v-for="[label, value] in [
                  ['运镜', o.camera],
                  ['主体', o.subject],
                  ['动作', o.action],
                  ['环境', o.setting],
                  ['光线', o.lighting],
                  ['色调', o.color],
                  ['节奏', o.motion_energy],
                  ['台词', o.dialogue],
                  ['音效', o.sfx],
                  ['转场', o.transition],
                  ['画面文字', o.on_screen_text],
                ]" :key="label">
                  <template v-if="value && value !== 'none'">
                    <dt>{{ label }}</dt>
                    <dd>{{ value }}</dd>
                  </template>
                </template>
              </dl>
            </div>
          </div>
        </template>

        <!-- 主体登记表 -->
        <template v-else-if="tab === 'subjects'">
          <div class="field-hint" style="margin-bottom: 12px">
            这些是 Pass1 跨镜头登记的主体。Ref2VA 模式下它们会依次变成
            &lt;Subject 1..N&gt;，并逐个写进 retention_analysis。
          </div>
          <div class="shot-list">
            <div v-for="(s, i) in store.subjects" :key="i" class="shot-card">
              <div class="shot-head">
                <span class="shot-no">&lt;Subject {{ i + 1 }}&gt;</span>
                <span class="mono faint">{{ s.label }}</span>
                <span v-if="s.kind" class="badge">{{ s.kind }}</span>
                <span v-if="s.shots.length" class="badge">
                  {{ s.shots.map((x) => `[Shot ${x}]`).join(' ') }}
                </span>
              </div>
              <dl class="shot-fields">
                <template v-for="[label, value] in [
                  ['外观', s.description],
                  ['需保持一致', s.notes],
                ]" :key="label">
                  <template v-if="value">
                    <dt>{{ label }}</dt>
                    <dd>{{ value }}</dd>
                  </template>
                </template>
              </dl>
            </div>
          </div>
        </template>

        <!-- 音频 -->
        <template v-else>
          <div v-if="!store.audio || !store.audio.has_audio" class="empty">
            该视频没有音轨，音频维度无法反推
          </div>
          <template v-else>
            <div class="row row-wrap" style="gap: 8px; margin-bottom: 14px">
              <span v-if="store.audio.mean_volume_db !== null" class="badge">
                平均 {{ store.audio.mean_volume_db.toFixed(1) }} dB
              </span>
              <span v-if="store.audio.peak_volume_db !== null" class="badge">
                峰值 {{ store.audio.peak_volume_db.toFixed(1) }} dB
              </span>
              <span v-if="store.audio.silence_ratio !== null" class="badge">
                静音 {{ (store.audio.silence_ratio * 100).toFixed(0) }}%
              </span>
              <span v-if="store.audio.note" class="badge info">{{ store.audio.note }}</span>
            </div>

            <div v-if="store.audio.segments.length" class="transcript">
              <div v-for="(seg, i) in store.audio.segments" :key="i" class="seg">
                <span class="mono faint seg-time">{{ fmtTime(seg.start) }}</span>
                <span>{{ seg.text }}</span>
              </div>
            </div>
            <div v-else-if="store.audio.transcript" class="prompt-box">
              {{ store.audio.transcript }}
            </div>
            <div v-else class="empty">未获得语音转写文本（可能无对白，或未配置 ASR）</div>
          </template>
        </template>
      </template>
    </div>
  </section>
</template>

<style scoped>
.anchors {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-bottom: 10px;
}

.anchor {
  padding: 3px 9px;
  border: 1px solid var(--border-soft);
  border-radius: 999px;
  background: var(--bg-soft);
  color: var(--text-dim);
  font-family: var(--mono);
  font-size: 11px;
  cursor: pointer;
  transition: color 0.14s, border-color 0.14s;
}

.anchor:hover {
  color: var(--accent-hover);
  border-color: var(--accent);
}

.stats-row {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  margin-top: 12px;
  font-size: 12px;
}

.shot-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
  max-height: 640px;
  overflow-y: auto;
  padding-right: 4px;
}

.shot-card {
  padding: 13px 15px;
  border: 1px solid var(--border-soft);
  border-radius: var(--radius);
  background: var(--bg-soft);
}

.shot-head {
  display: flex;
  align-items: center;
  gap: 9px;
  flex-wrap: wrap;
  margin-bottom: 10px;
}

.shot-no {
  font-size: 13px;
  font-weight: 600;
  color: var(--accent-hover);
}

.shot-fields {
  display: grid;
  grid-template-columns: 62px minmax(0, 1fr);
  gap: 5px 12px;
  margin: 0;
  font-size: 12.5px;
  line-height: 1.65;
}

.shot-fields dt {
  color: var(--text-faint);
  font-size: 11.5px;
  padding-top: 1px;
}

.shot-fields dd {
  margin: 0;
  color: var(--text-dim);
  word-break: break-word;
}

.transcript {
  max-height: 520px;
  overflow-y: auto;
  font-size: 12.5px;
  line-height: 1.8;
}

.seg {
  display: flex;
  gap: 12px;
  padding: 5px 0;
  border-bottom: 1px solid var(--border-soft);
}

.seg:last-child {
  border-bottom: none;
}

.seg-time {
  flex-shrink: 0;
  width: 56px;
}
</style>
