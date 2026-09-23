<script setup lang="ts">
import { computed, ref, watch } from 'vue'

import TrimBar from '@/components/TrimBar.vue'
import { useAnalyzeStore } from '@/stores/analyze'

const store = useAnalyzeStore()
const dragging = ref(false)
const fileInput = ref<HTMLInputElement | null>(null)
const showAdvanced = ref(false)
const showExtra = ref(false)

const ACCEPT = 'video/mp4,video/quicktime,video/x-matroska,video/webm,video/x-msvideo,video/*'

const pickedName = computed(() => store.file?.name ?? '')
const pickedSize = computed(() => {
  const s = store.file?.size ?? 0
  if (!s) return ''
  if (s < 1024 * 1024) return `${(s / 1024).toFixed(0)} KB`
  return `${(s / 1048576).toFixed(1)} MB`
})

/** 视频就绪后才能框选片段、才能开始反推。 */
const canStart = computed(() => {
  if (store.running || store.starting || store.preparing) return false
  return store.ready
})

/** 上传模式：选中文件就自动上传，传完就能预览和框选。 */
function onFileChosen(f: File) {
  store.setFile(f)
  store.resetSource()
  void store.prepare()
}

function pick() {
  fileInput.value?.click()
}

function onFileInput(e: Event) {
  const target = e.target as HTMLInputElement
  const f = target.files?.[0]
  if (f) onFileChosen(f)
  target.value = ''
}

function onDrop(e: DragEvent) {
  dragging.value = false
  const f = e.dataTransfer?.files?.[0]
  if (f) onFileChosen(f)
}

function onDragOver() {
  dragging.value = true
}

function onDragLeave() {
  dragging.value = false
}

/** 改了链接就把已下载的视频作废，避免「贴了新链接却推的是旧视频」。 */
watch(() => store.linkUrl, () => {
  if (store.ready && store.mode === 'link') store.resetSource()
})
</script>

<template>
  <section class="panel">
    <div class="panel-head">
      <div class="panel-title"><span class="dot" />选择视频来源</div>
      <div class="tabs">
        <button
          :class="['tab', { active: store.mode === 'upload' }]"
          @click="store.mode = 'upload'"
        >
          上传视频
        </button>
        <button
          :class="['tab', { active: store.mode === 'link' }]"
          @click="store.mode = 'link'"
        >
          B站 / 抖音链接
        </button>
      </div>
    </div>

    <div class="panel-body">
      <!-- ------------------------------------------------ 上传 -->
      <template v-if="store.mode === 'upload'">
        <div
          :class="['dropzone', { dragging, filled: !!store.file }]"
          @click="pick"
          @dragover.prevent="onDragOver"
          @dragleave="onDragLeave"
          @drop.prevent="onDrop"
        >
          <input
            ref="fileInput"
            type="file"
            :accept="ACCEPT"
            hidden
            @change="onFileInput"
          />

          <template v-if="store.file">
            <div class="dz-icon">🎬</div>
            <div class="dz-name">{{ pickedName }}</div>
            <div class="dz-meta faint">{{ pickedSize }} · 点击更换</div>
          </template>
          <template v-else>
            <div class="dz-icon">＋</div>
            <div class="dz-name">拖入视频，或点击选择</div>
            <div class="dz-meta faint">
              支持 mp4 / mov / mkv / webm / avi 等，单个最大
              {{ store.health ? '500' : '—' }}MB
            </div>
          </template>
        </div>

        <div v-if="store.uploading" class="upload-bar">
          <div class="progress-track">
            <div class="progress-fill" :style="{ width: store.uploadPercent + '%' }" />
          </div>
          <span class="faint mono">上传中 {{ store.uploadPercent }}%</span>
        </div>
      </template>

      <!-- ------------------------------------------------ 链接 -->
      <template v-else>
        <div class="field">
          <label class="field-label">视频链接或分享文案</label>
          <textarea
            v-model="store.linkUrl"
            class="textarea"
            rows="3"
            placeholder="直接粘贴抖音分享文案也行，会自动从中提取链接。例如：&#10;7.32 复制打开抖音，看看【某某】的作品 https://v.douyin.com/xxxxxx/"
          />
          <div class="field-hint">
            支持 B站（bilibili.com / b23.tv）与抖音（v.douyin.com）。
            抖音签名校验变动频繁，自动抓取不是总能成功，失败时请手动下载后上传——分析效果完全一致。
          </div>
        </div>

        <div class="row">
          <button class="btn btn-sm" :disabled="store.probing || !store.linkUrl.trim()" @click="store.probe()">
            <span v-if="store.probing" class="spinner" />
            {{ store.probing ? '解析中' : '先解析看看' }}
          </button>
        </div>

        <div v-if="store.probeResult" class="probe-card">
          <div class="row" style="gap: 8px; align-items: flex-start">
            <img
              v-if="store.probeResult.thumbnail"
              :src="store.probeResult.thumbnail"
              class="probe-thumb"
              alt=""
            />
            <div style="flex: 1; min-width: 0">
              <div class="probe-title">{{ store.probeResult.title || '（无标题）' }}</div>
              <div class="row row-wrap" style="margin-top: 6px; gap: 6px">
                <span class="badge">{{ store.probeResult.platform || '未知平台' }}</span>
                <span v-if="store.probeResult.uploader" class="badge">
                  {{ store.probeResult.uploader }}
                </span>
                <span v-if="store.probeResult.duration" class="badge">
                  {{ store.probeResult.duration.toFixed(1) }}s
                </span>
              </div>
              <div v-if="store.probeResult.note" class="field-hint" style="margin-top: 8px">
                {{ store.probeResult.note }}
              </div>
            </div>
          </div>
        </div>

        <div class="row" style="margin-top: 12px">
          <button
            class="btn"
            :disabled="store.preparing || !store.linkUrl.trim()"
            @click="store.prepare()"
          >
            <span v-if="store.preparing" class="spinner" />
            {{ store.preparing ? '下载中…' : store.ready ? '重新下载' : '下载视频' }}
          </button>
          <span class="faint" style="font-size: 12px">下载后可以预览并框选要反推的片段</span>
        </div>
      </template>

      <!-- ------------------------------------------------ 片段选择 -->
      <div v-if="store.ready && store.videoUrl" class="trim-section">
        <div class="trim-head">
          <span class="field-label">选择要反推的片段</span>
          <span class="faint" style="font-size: 12px">
            整片太长时，框出你要的那一段——无关的前后内容不会被写进提示词
          </span>
        </div>
        <TrimBar
          v-model="store.trimRange"
          :src="store.videoUrl"
          :duration="store.knownDuration"
        />
      </div>

      <!-- ------------------------------------------------ 画面说明 -->
      <div class="hint-section">
        <div class="trim-head">
          <span class="field-label">画面说明（可选，但很有用）</span>
          <span class="faint" style="font-size: 12px">
            静态帧判断不出「一镜到底还是多镜头切换」「这是什么作品/角色」「动作的前因后果」，
            你写一句就能大幅提升准确度
          </span>
        </div>
        <textarea
          v-model="store.options.content_hint"
          class="textarea"
          rows="3"
          placeholder="例如：一镜到底的跟拍运镜，全程没有切镜；主角是白发少女，穿黑色风衣；赛博朋克冷色调，霓虹反光；画面里在下雨"
        />
        <div class="field-hint">
          这段会同时喂给「观察」和「成文」两个阶段。写清<b>镜头结构</b>（一镜到底 / 多镜头切换）、
          <b>主体是谁</b>、<b>在做什么</b> 最有效。不要写「要生成什么样的视频」——那是下面「额外要求」的事。
        </div>
      </div>

      <!-- ------------------------------------------------ 模式 -->
      <div class="divider" />

      <div class="field">
        <label class="field-label">反推模式</label>
        <div class="mode-grid">
          <button
            v-for="m in store.primaryModes"
            :key="m.value"
            :class="['mode-card', { active: store.currentMode === m.value }]"
            @click="store.selectMode(m.value)"
          >
            <div class="mode-name">{{ m.label }}</div>
            <div class="mode-desc">{{ m.description }}</div>
          </button>
        </div>

        <!-- H3 有两个变体：T2VA 要自洽、Ref2VA 要出参考标签，写法不同，必须显式选 -->
        <div v-if="store.currentVariants.length > 1" class="variants">
          <button
            v-for="v in store.currentVariants"
            :key="v.value"
            :class="['variant', { active: store.options.format === v.value }]"
            @click="store.selectFormat(v.value)"
          >
            {{ v.label }}
          </button>
        </div>
        <div v-if="store.formatInfo?.variant.description" class="field-hint">
          {{ store.formatInfo.variant.description }}
        </div>

        <button
          v-if="store.extraModes.length"
          class="btn btn-ghost btn-sm"
          style="margin-top: 10px"
          @click="showExtra = !showExtra"
        >
          {{ showExtra ? '收起' : '其他格式' }}
        </button>
        <div v-if="showExtra" class="variants" style="margin-top: 10px">
          <button
            v-for="m in store.extraModes"
            :key="m.value"
            :class="['variant', { active: store.options.format === m.variants[0]?.value }]"
            @click="store.selectMode(m.value)"
          >
            {{ m.label }}
          </button>
        </div>
      </div>

      <div class="row row-wrap" style="gap: 18px; margin-bottom: 14px">
        <label class="checkbox">
          <input v-model="store.options.enable_asr" type="checkbox" />
          语音转写（台词 / 口型 / 音效必须靠它）
        </label>
        <label class="checkbox">
          <input v-model="store.options.enable_scene_split" type="checkbox" />
          镜头自动切分
        </label>
        <label class="checkbox">
          <input v-model="store.options.language" type="checkbox" true-value="zh" false-value="en" />
          提示词用中文
        </label>
      </div>

      <button class="btn btn-ghost btn-sm" @click="showAdvanced = !showAdvanced">
        {{ showAdvanced ? '收起' : '展开' }}高级选项
      </button>

      <div v-if="showAdvanced" class="advanced">
        <div class="field">
          <label class="field-label">抽帧总预算（帧）</label>
          <input
            v-model.number="store.options.max_total_frames"
            class="input"
            type="number"
            min="4"
            max="800"
            :placeholder="`留空用服务端默认（${store.settings?.frames.max_total_frames ?? 96}）`"
          />
          <div class="field-hint">
            <strong>这只是安全网</strong>，不是目标值 —— 实际抽多少由镜头时长决定
            （每镜每秒约一帧，上限
            {{ store.settings?.frames.max_frames_per_shot ?? 8 }} 帧）。
            15 秒的短片只会用到 15 帧左右，调大不影响短片，只防止长视频失控。
            <br />
            一帧约 1100 tokens。128k 上下文用 96，256k 用 200，1M 可以用到 400。
          </div>
        </div>

        <div class="field">
          <label class="field-label">镜头内取帧间隔（秒）</label>
          <input
            v-model.number="store.options.frame_interval_seconds"
            class="input"
            type="number"
            step="0.1"
            min="0.2"
            max="5"
            :placeholder="`留空用服务端默认（${store.settings?.frames.frame_interval_seconds ?? 1}）`"
          />
          <div class="field-hint">
            越小越密。1.0 表示 8 秒的镜头取 8 帧。想更细就调到 0.5。
          </div>
        </div>

        <div class="field">
          <label class="field-label">提示词词数上限</label>
          <input
            v-model.number="store.options.prompt_word_limit"
            class="input"
            type="number"
            min="200"
            max="3000"
            :placeholder="`留空用服务端默认（${store.settings?.prompt.word_limit ?? 700}）`"
          />
          <div class="field-hint">
            限的是<strong>整篇</strong>（含主体定义和保留分析），不只是正文。
            超了会自动压缩一次。你用的视频模型吃得下多长就写多长。
          </div>
        </div>

        <div class="field">
          <label class="field-label">额外要求（会拼进最终提示词任务里）</label>
          <textarea
            v-model="store.options.extra_instruction"
            class="textarea"
            rows="2"
            placeholder="例如：重点描述运镜与光线；忽略背景人群；强调服装材质"
          />
        </div>

        <div class="field">
          <label class="field-label">目标时长（秒，可留空）</label>
          <input
            v-model.number="store.options.target_duration"
            class="input"
            type="number"
            min="1"
            placeholder="生成视频的期望时长，留空则按原片时长"
          />
        </div>
      </div>

      <!-- ------------------------------------------------ 启动 -->
      <div class="row" style="margin-top: 18px; gap: 12px">
        <button class="btn btn-primary btn-lg" :disabled="!canStart" @click="store.start()">
          <span v-if="store.starting" class="spinner" />
          {{ store.starting ? '反推中…' : '开始反推' }}
        </button>
        <span v-if="!store.ready && !store.preparing" class="faint" style="font-size: 12px">
          {{ store.mode === 'upload' ? '先选一个视频文件' : '先粘贴链接并下载' }}
        </span>
        <button
          v-if="store.job || store.error"
          class="btn btn-ghost"
          @click="store.reset()"
        >
          清空
        </button>
      </div>
    </div>
  </section>
</template>

<style scoped>
.dropzone {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 4px;
  min-height: 150px;
  padding: 24px;
  border: 1.5px dashed var(--border);
  border-radius: var(--radius);
  background: var(--bg-soft);
  cursor: pointer;
  transition: border-color 0.16s, background 0.16s;
}

.dropzone:hover,
.dropzone.dragging {
  border-color: var(--accent);
  background: var(--accent-soft);
}

.dropzone.filled {
  border-style: solid;
  border-color: rgba(109, 139, 255, 0.4);
}

.dz-icon {
  font-size: 24px;
  line-height: 1;
  margin-bottom: 4px;
}

.dz-name {
  font-size: 13.5px;
  font-weight: 500;
  text-align: center;
  word-break: break-all;
}

.dz-meta {
  font-size: 11.5px;
  text-align: center;
}

.upload-bar {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-top: 12px;
}

.upload-bar .progress-track {
  flex: 1;
}

.divider {
  height: 1px;
  background: var(--border-soft);
  margin: 18px 0;
}

.probe-card {  margin-top: 14px;
  padding: 12px;
  border: 1px solid var(--border-soft);
  border-radius: var(--radius);
  background: var(--bg-soft);
}

.probe-thumb {
  width: 96px;
  height: 56px;
  object-fit: cover;
  border-radius: var(--radius-sm);
  flex-shrink: 0;
  background: var(--panel-2);
}

.probe-title {
  font-size: 13px;
  font-weight: 500;
  line-height: 1.5;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.advanced {
  margin-top: 14px;
  padding: 14px;
  border: 1px solid var(--border-soft);
  border-radius: var(--radius);
  background: var(--bg-soft);
}

/* 两张大卡片选模式。H3 与 Seedance 是并列的两种产物，不给下拉框——
   下拉框会把「这是两个不同目标模型」这件事藏起来。 */
.mode-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
}

.mode-card {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 12px 14px;
  text-align: left;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--bg-soft);
  color: var(--text);
  cursor: pointer;
  transition: border-color 0.16s, background 0.16s;
  font-family: inherit;
}

.mode-card:hover {
  border-color: var(--border-hover, #3a4460);
}

.mode-card.active {
  border-color: var(--accent);
  background: var(--accent-soft);
}

.mode-name {
  font-size: 13.5px;
  font-weight: 500;
}

.mode-card.active .mode-name {
  color: var(--accent-hover);
}

.mode-desc {
  font-size: 11.5px;
  line-height: 1.5;
  color: var(--text-faint);
  display: -webkit-box;
  -webkit-line-clamp: 3;
  line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.variants {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 12px;
}

.variant {
  padding: 6px 12px;
  font-size: 12px;
  font-family: inherit;
  border: 1px solid var(--border);
  border-radius: 999px;
  background: transparent;
  color: var(--text-dim);
  cursor: pointer;
  transition: border-color 0.16s, color 0.16s, background 0.16s;
}

.variant:hover {
  color: var(--text);
}

.variant.active {
  border-color: var(--accent);
  background: var(--accent-soft);
  color: var(--accent-hover);
}

.trim-section {
  margin-top: 18px;
  padding-top: 16px;
  border-top: 1px solid var(--border-soft);
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.trim-head {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.hint-section {
  margin-top: 18px;
  padding-top: 16px;
  border-top: 1px solid var(--border-soft);
  display: flex;
  flex-direction: column;
  gap: 10px;
}
</style>
