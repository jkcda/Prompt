/** 反推流程的全局状态机。
 *
 *  空闲 → 选源（上传 / 链接）→ 创建任务 → SSE 推进度 → 拉结果 → 展示
 *
 *  进度用 SSE 实时更新；任务结束后再 GET 一次详情拿完整结果（SSE 只传轻量事件）。
 */

import { defineStore } from 'pinia'
import { computed, ref } from 'vue'

import * as api from '@/api'
import { errorMessage } from '@/api'
import type {
  AnalyzeOptions,
  HealthInfo,
  Job,
  JobEvent,
  JobSummary,
  ProbeResult,
  PromptFormat,
  PromptMode,
  PromptModeOption,
  UploadResult,
} from '@/types'

export const useAnalyzeStore = defineStore('analyze', () => {
  // ---------------- 环境 ----------------
  const health = ref<HealthInfo | null>(null)
  const modes = ref<PromptModeOption[]>([])
  const bootError = ref('')

  // ---------------- 选项 ----------------
  const options = ref<AnalyzeOptions>({
    format: 'h3',
    language: 'en',
    enable_asr: true,
    enable_scene_split: true,
    max_total_frames: null,
    extra_instruction: '',
    target_duration: null,
  })

  /** 只用于主选择位的两种模式（H3 / Seedance）。 */
  const primaryModes = computed(() => modes.value.filter((m) => m.primary))
  /** 兜底格式，界面上折进「其他格式」。 */
  const extraModes = computed(() => modes.value.filter((m) => !m.primary))

  /** 当前 format 属于哪个模式。 */
  const currentMode = computed<PromptMode | ''>(() => {
    const hit = modes.value.find((m) => m.variants.some((v) => v.value === options.value.format))
    return hit?.value ?? ''
  })

  /** 当前模式下的变体；单变体模式返回空数组，界面不显示切换器。 */
  const currentVariants = computed(() => {
    const hit = modes.value.find((m) => m.value === currentMode.value)
    if (!hit || hit.variants.length < 2) return []
    return hit.variants
  })

  /** 当前 format 的展示信息（任意模式，含兜底格式）。 */
  const formatInfo = computed(() => {
    for (const m of modes.value) {
      const v = m.variants.find((x) => x.value === options.value.format)
      if (v) return { mode: m, variant: v }
    }
    return null
  })

  /**
   * 切换模式。选中的是变体而不是模式本身——模式只是分组。
   * 切到某个模式时，用它的默认变体。
   */
  function selectMode(mode: PromptMode) {
    const hit = modes.value.find((m) => m.value === mode)
    if (!hit || !hit.variants.length) return
    const preferred = hit.variants.find((v) => v.default) ?? hit.variants[0]
    options.value.format = preferred.value
  }

  function selectFormat(format: PromptFormat) {
    options.value.format = format
  }

  // ---------------- 来源 ----------------
  const mode = ref<'upload' | 'link'>('upload')
  const file = ref<File | null>(null)
  const localPreviewUrl = ref('')
  const uploadResult = ref<UploadResult | null>(null)
  const uploadPercent = ref(0)
  const uploading = ref(false)

  const linkUrl = ref('')
  const probeResult = ref<ProbeResult | null>(null)
  const probing = ref(false)

  // ---------------- 任务 ----------------
  const jobId = ref('')
  const job = ref<Job | null>(null)
  const events = ref<JobEvent[]>([])
  const starting = ref(false)
  const error = ref('')
  const elapsed = ref(0)

  const history = ref<JobSummary[]>([])
  const historyLoading = ref(false)

  let unsubscribe: (() => void) | null = null
  let timer: number | null = null

  // ---------------- 派生 ----------------
  const running = computed(
    () => starting.value || (job.value?.state === 'running' || job.value?.state === 'pending'),
  )

  const progress = computed(() => job.value?.progress.percent ?? 0)
  const stageLabel = computed(() => job.value?.progress.stage_label ?? '')
  const stageMessage = computed(() => job.value?.progress.message ?? '')

  /** 结果页要显示的视频地址：优先任务的，其次本地预览。 */
  const videoUrl = computed(
    () => job.value?.video_url || uploadResult.value?.video_url || localPreviewUrl.value || '',
  )

  const prompt = computed(() => job.value?.result?.prompt ?? '')
  const observations = computed(() => job.value?.result?.observations ?? [])
  const subjects = computed(() => job.value?.result?.subjects ?? [])
  const frameUrls = computed(() => job.value?.result?.frame_urls ?? [])
  const shots = computed(() => job.value?.result?.shots ?? [])
  const audio = computed(() => job.value?.result?.audio ?? null)
  const media = computed(() => job.value?.result?.media ?? null)
  const stats = computed(() => job.value?.result?.stats ?? {})

  /** 后端给的阶段中文名，用于进度条上方。 */
  const stageHint = computed(() => {
    if (job.value?.error) return '失败'
    if (job.value?.state === 'succeeded') return '完成'
    return stageMessage.value || stageLabel.value
  })

  // ---------------- 初始化 ----------------

  async function init() {
    bootError.value = ''
    try {
      const [h, m] = await Promise.all([api.getHealth(), api.getFormats()])
      health.value = h
      modes.value = m
      // 后端换了默认变体时跟随，避免本地写死的 format 在后端已下线
      const known = m.some((x) => x.variants.some((v) => v.value === options.value.format))
      if (!known) {
        const first = m.find((x) => x.primary) ?? m[0]
        const fallback = first?.variants.find((v) => v.default) ?? first?.variants[0]
        if (fallback) options.value.format = fallback.value
      }
    } catch (e) {
      bootError.value = errorMessage(e)
    }
    void loadHistory()
  }

  async function loadHistory() {
    historyLoading.value = true
    try {
      history.value = await api.listJobs(30)
    } catch {
      /* 历史加载失败不阻塞主流程 */
    } finally {
      historyLoading.value = false
    }
  }

  // ---------------- 选源 ----------------

  function setFile(f: File | null) {
    clearPreview()
    file.value = f
    uploadResult.value = null
    if (f) {
      localPreviewUrl.value = URL.createObjectURL(f)
    }
  }

  function clearPreview() {
    if (localPreviewUrl.value) {
      URL.revokeObjectURL(localPreviewUrl.value)
      localPreviewUrl.value = ''
    }
  }

  async function probe() {
    const url = linkUrl.value.trim()
    if (!url) {
      error.value = '请先粘贴视频链接或分享文案'
      return
    }
    probing.value = true
    error.value = ''
    probeResult.value = null
    try {
      probeResult.value = await api.probeLink(url)
    } catch (e) {
      error.value = errorMessage(e)
    } finally {
      probing.value = false
    }
  }

  // ---------------- 启动 ----------------

  async function start() {
    error.value = ''
    events.value = []
    job.value = null
    jobId.value = ''
    starting.value = true

    try {
      let id = ''
      if (mode.value === 'upload') {
        if (!file.value) throw new Error('请先选择视频文件')

        uploading.value = true
        uploadPercent.value = 0
        const up = await api.uploadVideo(file.value, (p) => (uploadPercent.value = p))
        uploadResult.value = up
        uploading.value = false

        const res = await api.startAnalyze(up.file_id, up.name, options.value)
        id = res.job_id
      } else {
        if (!linkUrl.value.trim()) throw new Error('请先粘贴视频链接')
        const res = await api.startFetch(linkUrl.value.trim(), options.value)
        id = res.job_id
      }

      jobId.value = id
      await refreshJob()
      attach(id)
    } catch (e) {
      error.value = errorMessage(e)
      uploading.value = false
    } finally {
      starting.value = false
    }
  }

  // ---------------- SSE ----------------

  function attach(id: string) {
    detach()
    startTimer()
    unsubscribe = api.subscribeJob(
      id,
      (e) => {
        events.value.push(e)

        // 轻量事件直接更新进度，不必每次都拉全量
        if (e.type === 'progress' && job.value) {
          job.value.progress = {
            stage: e.stage ?? '',
            stage_label: e.stage_label ?? '',
            percent: e.percent ?? 0,
            message: e.message ?? '',
          }
          job.value.state = 'running'
        }
        if (e.type === 'video_ready' && e.video_url && job.value) {
          job.value.video_url = e.video_url
        }
        if (e.type === 'fetched' && job.value) {
          if (e.title) job.value.title = e.title
        }
        if (e.type === 'done' || e.type === '__close__') {
          void refreshJob().then(() => {
            stopTimer()
            detach()
            void loadHistory()
          })
        }
      },
      () => {
        // 连接异常时兜底拉一次，避免 UI 卡在旧进度
        void refreshJob()
      },
    )
  }

  function detach() {
    if (unsubscribe) {
      unsubscribe()
      unsubscribe = null
    }
  }

  function startTimer() {
    stopTimer()
    elapsed.value = 0
    timer = window.setInterval(() => (elapsed.value += 1), 1000)
  }

  function stopTimer() {
    if (timer !== null) {
      window.clearInterval(timer)
      timer = null
    }
  }

  // ---------------- 任务读取 ----------------

  async function refreshJob() {
    if (!jobId.value) return
    try {
      const fresh = await api.getJob(jobId.value)
      job.value = fresh
      if (fresh.state === 'succeeded' || fresh.state === 'failed' || fresh.state === 'cancelled') {
        stopTimer()
      }
    } catch (e) {
      error.value = errorMessage(e)
    }
  }

  async function loadJob(id: string) {
    detach()
    stopTimer()
    error.value = ''
    events.value = []
    jobId.value = id
    await refreshJob()
    const state = job.value?.state
    if (state === 'running' || state === 'pending') {
      attach(id)
    }
  }

  async function cancel() {
    if (!jobId.value) return
    try {
      await api.cancelJob(jobId.value)
      await refreshJob()
    } catch (e) {
      error.value = errorMessage(e)
    }
  }

  async function remove(id: string) {
    try {
      await api.deleteJob(id)
      if (jobId.value === id) reset()
      await loadHistory()
    } catch (e) {
      error.value = errorMessage(e)
    }
  }

  function reset() {
    detach()
    stopTimer()
    clearPreview()
    file.value = null
    uploadResult.value = null
    uploadPercent.value = 0
    linkUrl.value = ''
    probeResult.value = null
    jobId.value = ''
    job.value = null
    events.value = []
    error.value = ''
    elapsed.value = 0
  }

  function clearError() {
    error.value = ''
  }

  return {
    // 环境
    health, modes, primaryModes, extraModes, bootError,
    // 模式与格式
    currentMode, currentVariants, formatInfo, selectMode, selectFormat,
    // 选项
    options,
    // 来源
    mode, file, localPreviewUrl, uploadResult, uploadPercent, uploading,
    linkUrl, probeResult, probing,
    // 任务
    jobId, job, events, starting, error, elapsed, history, historyLoading,
    // 派生
    running, progress, stageLabel, stageMessage, stageHint,
    videoUrl, prompt, observations, subjects, frameUrls, shots, audio, media, stats,
    // 动作
    init, loadHistory, setFile, probe, start, refreshJob, loadJob, cancel, remove,
    reset, clearError,
  }
})
