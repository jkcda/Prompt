<script setup lang="ts">
import { computed, ref, watch } from 'vue'

import { errorMessage, getSettings, listModels, testModelConfig, updateSettings } from '@/api'
import { useAnalyzeStore } from '@/stores/analyze'
import type { ModelTestResult, SettingsInfo } from '@/types'

const store = useAnalyzeStore()

const props = defineProps<{ open: boolean }>()
const emit = defineEmits<{ close: []; saved: [] }>()

/** 常见服务商的 OpenAI 兼容地址。填错地址是最常见的失败原因，给预设省事。 */
const PRESETS = [
  { name: 'ModelScope 魔搭', url: 'https://api-inference.modelscope.cn/v1', note: '国内直连，免费额度' },
  { name: '阿里百炼', url: 'https://dashscope.aliyuncs.com/compatible-mode/v1', note: 'Qwen 全系' },
  { name: '火山方舟', url: 'https://ark.cn-beijing.volces.com/api/v3', note: '豆包系列' },
  { name: 'OpenRouter', url: 'https://openrouter.ai/api/v1', note: '聚合，需外网' },
  { name: '本地 vLLM', url: 'http://127.0.0.1:8001/v1', note: '自建推理' },
]

const loading = ref(false)
const saving = ref(false)
const testing = ref(false)
const listing = ref(false)
const error = ref('')
const notice = ref('')
const settings = ref<SettingsInfo | null>(null)

const baseUrl = ref('')
const model = ref('')
const apiKey = ref('')
const audioInput = ref(false)

const candidates = ref<string[]>([])
const listNote = ref('')
const testingModel = ref('')
/** 逐个模型的测试结果，key 是模型名。 */
const results = ref<Record<string, ModelTestResult>>({})

const keyPlaceholder = computed(
  () => settings.value?.vlm.api_key_masked || 'sk-... 或 ms-...',
)
const keyTouched = computed(() => apiKey.value.trim().length > 0)

const dirty = computed(() => {
  if (!settings.value) return false
  const v = settings.value.vlm
  return (
    baseUrl.value.trim() !== v.base_url ||
    model.value.trim() !== v.model ||
    audioInput.value !== v.audio_input ||
    keyTouched.value
  )
})

const currentResult = computed(() => results.value[model.value.trim()] ?? null)

watch(
  () => props.open,
  async (open) => {
    if (!open) return
    error.value = ''
    notice.value = ''
    candidates.value = []
    listNote.value = ''
    results.value = {}
    loading.value = true
    try {
      const s = await getSettings()
      settings.value = s
      baseUrl.value = s.vlm.base_url
      model.value = s.vlm.model
      audioInput.value = s.vlm.audio_input
      apiKey.value = ''
    } catch (e) {
      error.value = errorMessage(e)
    } finally {
      loading.value = false
    }
  },
)

function usePreset(url: string) {
  baseUrl.value = url
}

async function fetchModels() {
  listing.value = true
  error.value = ''
  listNote.value = ''
  try {
    const r = await listModels(baseUrl.value.trim())
    candidates.value = r.models
    listNote.value = r.ok
      ? `${r.message}（列表不代表账号真实可用，请逐个测）`
      : r.message
  } catch (e) {
    error.value = errorMessage(e)
  } finally {
    listing.value = false
  }
}

/** 测试某个模型。不传就用当前输入框里的配置。 */
async function testOne(target?: string) {
  const name = (target ?? model.value).trim()
  if (!name) return
  testing.value = true
  testingModel.value = name
  error.value = ''
  try {
    const r = await testModelConfig({
      model: name,
      base_url: baseUrl.value.trim(),
      // 留空 = 沿用已保存的 key，所以只想换模型时不用重填
      api_key: keyTouched.value ? apiKey.value.trim() : undefined,
    })
    results.value = { ...results.value, [name]: r }
    if (!target && r.vision) {
      notice.value = `${name} 可以读图，保存后即生效`
    }
  } catch (e) {
    error.value = errorMessage(e)
  } finally {
    testing.value = false
    testingModel.value = ''
  }
}

function pick(name: string) {
  model.value = name
  notice.value = ''
}

async function save() {
  if (!settings.value) return
  saving.value = true
  error.value = ''
  notice.value = ''
  try {
    const patch: Record<string, unknown> = {
      VLM_BASE_URL: baseUrl.value.trim(),
      VLM_MODEL: model.value.trim(),
      VLM_AUDIO_INPUT: audioInput.value ? 'true' : 'false',
    }
    // 只在用户真的改了 key 时才写，避免把脱敏值写回去覆盖真 key
    if (keyTouched.value) patch.VLM_API_KEY = apiKey.value.trim()

    const r = await updateSettings(patch)
    if (!r.ok) {
      error.value = r.message || '保存失败'
      return
    }
    apiKey.value = ''
    settings.value = await getSettings()
    notice.value = '已保存，下一次反推立即生效'
    emit('saved')
    // 让头部的模型徽标跟着更新
    await store.init()
  } catch (e) {
    error.value = errorMessage(e)
  } finally {
    saving.value = false
  }
}

function verdictClass(r: ModelTestResult | null): string {
  if (!r) return ''
  if (r.vision) return 'ok'
  if (r.inconclusive) return 'warn'
  return 'err'
}

function verdictText(r: ModelTestResult | null): string {
  if (!r) return ''
  if (r.vision) return '能读图'
  if (r.inconclusive) return '无法判定'
  return r.ok ? '通但读不到图' : '不可用'
}
</script>

<template>
  <div v-if="open" class="overlay" @click.self="emit('close')">
    <div class="sheet">
      <div class="sheet-head">
        <div>
          <div class="sheet-title">模型设置</div>
          <div class="faint" style="font-size: 12px; margin-top: 2px">
            改完保存即刻生效，不用重启服务
          </div>
        </div>
        <button class="btn btn-ghost btn-sm" @click="emit('close')">关闭</button>
      </div>

      <div class="sheet-body">
        <div v-if="loading" class="empty"><span class="spinner" /> 读取配置…</div>

        <template v-else-if="settings">
          <!-- ---------------------------------------- 服务商 -->
          <div class="field">
            <label class="field-label">服务商地址（OpenAI 兼容）</label>
            <div class="presets">
              <button
                v-for="p in PRESETS"
                :key="p.url"
                :class="['chip', { active: baseUrl.trim() === p.url }]"
                :title="p.note"
                @click="usePreset(p.url)"
              >
                {{ p.name }}
              </button>
            </div>
            <input v-model="baseUrl" class="input" placeholder="https://.../v1" />
            <div class="field-hint">带不带 <code>/v1</code> 都行，代码会自动补。</div>
          </div>

          <!-- ---------------------------------------- Key -->
          <div class="field">
            <label class="field-label">API Key</label>
            <input
              v-model="apiKey"
              class="input"
              type="password"
              autocomplete="off"
              :placeholder="keyPlaceholder"
            />
            <div class="field-hint">
              <template v-if="settings.vlm.configured && !keyTouched">
                已配置（{{ settings.vlm.api_key_masked }}）。留空表示不改。
              </template>
              <template v-else-if="!keyTouched"> 还没配，填一个才能用。 </template>
              <template v-else>将替换为新的 key。</template>
            </div>
          </div>

          <!-- ---------------------------------------- 模型 -->
          <div class="field">
            <label class="field-label">模型</label>
            <div class="row" style="gap: 8px">
              <input v-model="model" class="input" style="flex: 1" placeholder="Vendor/Model-Name" />
              <button class="btn btn-sm" :disabled="listing" @click="fetchModels">
                <span v-if="listing" class="spinner" />
                {{ listing ? '拉取中' : '拉取列表' }}
              </button>
              <button
                class="btn btn-sm"
                :disabled="testing || !model.trim()"
                @click="testOne()"
              >
                <span v-if="testing && !testingModel" class="spinner" />
                {{ testing && !testingModel ? '测试中' : '测试' }}
              </button>
            </div>

            <div
              v-if="currentResult"
              :class="['verdict', verdictClass(currentResult)]"
            >
              <strong>{{ verdictText(currentResult) }}</strong>
              <span>{{ currentResult.message }}</span>
            </div>

            <div class="field-hint">
              测试会真的发一张纯红图，回答里出现 red 才算「能读图」——
              只发文本的话，纯文本模型也能通过，你会以为配好了。
            </div>
          </div>

          <!-- ---------------------------------------- 候选列表 -->
          <div v-if="candidates.length" class="field">
            <label class="field-label">候选模型（{{ candidates.length }}）</label>
            <div v-if="listNote" class="field-hint" style="margin-bottom: 8px">
              {{ listNote }}
            </div>
            <div class="model-list">
              <div
                v-for="m in candidates"
                :key="m"
                :class="['model-row', { picked: m === model.trim() }]"
              >
                <button class="model-name mono" @click="pick(m)">{{ m }}</button>
                <span
                  v-if="results[m]"
                  :class="['badge', verdictClass(results[m])]"
                >
                  {{ verdictText(results[m]) }}
                </span>
                <button
                  class="btn btn-sm"
                  :disabled="testing"
                  @click="testOne(m)"
                >
                  <span v-if="testingModel === m" class="spinner" />
                  {{ testingModel === m ? '…' : '测' }}
                </button>
              </div>
            </div>
            <div class="field-hint">
              列表里的模型**不代表能用** —— 很多服务商只返回精选列表，
              有些模型没部署推理服务，调用会报 has no provider supported。
              所以逐个点「测」，绿灯的才是真能用。
            </div>
          </div>

          <!-- ---------------------------------------- 音频 -->
          <div class="field">
            <label class="checkbox">
              <input v-model="audioInput" type="checkbox" />
              把音频直接附给模型（`VLM_AUDIO_INPUT`）
            </label>
            <div class="field-hint">
              ⚠ 能收音频的模型很少，很多「多模态」模型只支持图片。
              开启后如果模型不支持，反推会报错 —— 那时的提示会告诉你关掉它。
              建议先用「测试」确认能读图，再考虑这个。
            </div>
          </div>

          <!-- ---------------------------------------- 只读信息 -->
          <div class="divider" />
          <div class="facts">
            <div><span class="faint">ffmpeg</span> <span class="mono">{{ settings.ffmpeg.path || '未找到' }}</span></div>
            <div><span class="faint">ffprobe</span> <span class="mono">{{ settings.ffmpeg.ffprobe || '缺失（用 ffmpeg -i 替代）' }}</span></div>
            <div><span class="faint">语音转写</span> <span class="mono">{{ settings.asr.configured ? '已配置' : '未配置' }}</span></div>
            <div><span class="faint">抽帧预算</span> <span class="mono">{{ settings.frames.max_total_frames }} 帧 / 每镜最多 {{ settings.frames.max_frames_per_shot }}</span></div>
            <div><span class="faint">数据库</span> <span class="mono">{{ settings.database.path }}</span></div>
          </div>
        </template>

        <div v-if="error" class="alert err">{{ error }}</div>
        <div v-if="notice" class="alert ok">{{ notice }}</div>
      </div>

      <div class="sheet-foot">
        <span v-if="dirty" class="faint" style="font-size: 12px">有未保存的改动</span>
        <span v-else class="faint" style="font-size: 12px">配置已是最新</span>
        <div class="row" style="gap: 8px">
          <button class="btn btn-ghost btn-sm" @click="emit('close')">取消</button>
          <button
            class="btn btn-primary btn-sm"
            :disabled="saving || !settings"
            @click="save"
          >
            <span v-if="saving" class="spinner" />
            {{ saving ? '保存中' : '保存' }}
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.overlay {
  position: fixed;
  inset: 0;
  z-index: 60;
  background: rgba(4, 6, 10, 0.72);
  display: flex;
  align-items: flex-start;
  justify-content: center;
  padding: 40px 20px;
  overflow-y: auto;
}

.sheet {
  width: 100%;
  max-width: 680px;
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow);
  display: flex;
  flex-direction: column;
  max-height: calc(100vh - 80px);
}

.sheet-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  padding: 16px 20px;
  border-bottom: 1px solid var(--border-soft);
}

.sheet-title {
  font-size: 15px;
  font-weight: 500;
}

.sheet-body {
  padding: 18px 20px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.sheet-foot {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 13px 20px;
  border-top: 1px solid var(--border-soft);
}

.presets {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-bottom: 8px;
}

.chip {
  padding: 5px 11px;
  font-size: 12px;
  font-family: inherit;
  border: 1px solid var(--border);
  border-radius: 999px;
  background: transparent;
  color: var(--text-dim);
  cursor: pointer;
  transition: border-color 0.16s, color 0.16s, background 0.16s;
}

.chip:hover {
  color: var(--text);
}

.chip.active {
  border-color: var(--accent);
  background: var(--accent-soft);
  color: var(--accent-hover);
}

.verdict {
  margin-top: 10px;
  padding: 9px 12px;
  border-radius: var(--radius-sm);
  font-size: 12px;
  line-height: 1.6;
  border: 1px solid var(--border);
  background: var(--bg-soft);
  display: flex;
  gap: 8px;
  align-items: baseline;
}

.verdict.ok {
  border-color: rgba(62, 207, 142, 0.4);
  background: rgba(62, 207, 142, 0.08);
}

.verdict.warn {
  border-color: rgba(240, 184, 73, 0.4);
  background: rgba(240, 184, 73, 0.08);
}

.verdict.err {
  border-color: rgba(255, 107, 107, 0.4);
  background: rgba(255, 107, 107, 0.08);
}

.verdict strong {
  font-weight: 500;
  white-space: nowrap;
}

.model-list {
  max-height: 220px;
  overflow-y: auto;
  border: 1px solid var(--border-soft);
  border-radius: var(--radius);
  background: var(--bg-soft);
}

.model-row {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 7px 10px;
  border-bottom: 1px solid var(--border-soft);
}

.model-row:last-child {
  border-bottom: none;
}

.model-row.picked {
  background: var(--accent-soft);
}

.model-name {
  flex: 1;
  min-width: 0;
  text-align: left;
  font-size: 12px;
  font-family: var(--mono);
  background: none;
  border: none;
  color: var(--text-dim);
  cursor: pointer;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  padding: 2px 0;
}

.model-name:hover {
  color: var(--accent-hover);
}

.model-row.picked .model-name {
  color: var(--accent-hover);
}

.facts {
  display: flex;
  flex-direction: column;
  gap: 5px;
  font-size: 12px;
  line-height: 1.6;
}

.facts .mono {
  word-break: break-all;
}

.alert {
  padding: 10px 12px;
  border-radius: var(--radius-sm);
  font-size: 12px;
  line-height: 1.6;
  white-space: pre-wrap;
}

.alert.err {
  border: 1px solid rgba(255, 107, 107, 0.4);
  background: rgba(255, 107, 107, 0.08);
  color: #ffb3b3;
}

.alert.ok {
  border: 1px solid rgba(62, 207, 142, 0.4);
  background: rgba(62, 207, 142, 0.08);
  color: #9be8c4;
}

/* .divider 只在 SourcePicker 里定义过（scoped），这里要自己来一份 */
.divider {
  height: 1px;
  background: var(--border-soft);
}
</style>
