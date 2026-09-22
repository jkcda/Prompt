/** 后端 API 客户端。 */

import axios, { AxiosError } from 'axios'
import type {
  AnalyzeOptions,
  PromptModeOption,
  HealthInfo,
  Job,
  JobEvent,
  JobSummary,
  ModelListResult,
  ModelTestResult,
  ProbeResult,
  SettingsInfo,
  UploadResult,
} from '@/types'

export const http = axios.create({
  baseURL: '/api',
  timeout: 60_000,
})

/** 把后端 `detail` 抽成可读的错误消息。 */
export function errorMessage(err: unknown): string {
  if (err instanceof AxiosError) {
    const detail = err.response?.data?.detail
    if (typeof detail === 'string') return detail
    if (Array.isArray(detail)) {
      return detail.map((d: { msg?: string }) => d?.msg ?? JSON.stringify(d)).join('；')
    }
    if (err.code === 'ECONNABORTED') return '请求超时'
    if (!err.response) return '无法连接后端，请确认服务已启动'
    return `HTTP ${err.response.status}`
  }
  if (err instanceof Error) return err.message
  return String(err)
}

// ---------------------------------------------------------------- 系统

export const getHealth = () => http.get<HealthInfo>('/health').then((r) => r.data)

export const checkVLM = () =>
  http.get<ModelTestResult>('/health/vlm').then((r) => r.data)

/** 试一组模型配置但**不保存**。api_key 留空表示沿用已保存的。 */
export const testModelConfig = (payload: {
  model?: string
  base_url?: string
  api_key?: string
}) => http.post<ModelTestResult>('/health/vlm', payload, { timeout: 180000 }).then((r) => r.data)

/** 拉服务商声明的模型列表。注意这不代表账号真实可用范围，要逐个测。 */
export const listModels = (baseUrl?: string) =>
  http.get<ModelListResult>('/models', { params: baseUrl ? { base_url: baseUrl } : {} })
    .then((r) => r.data)

export const getFormats = () => http.get<PromptModeOption[]>('/formats').then((r) => r.data)

export const getSettings = () => http.get<SettingsInfo>('/settings').then((r) => r.data)

export const updateSettings = (patch: Record<string, unknown>) =>
  http.post<{ ok: boolean; updated?: string[]; message?: string }>('/settings', patch)
    .then((r) => r.data)

// ---------------------------------------------------------------- 上传 / 反推

export const uploadVideo = (
  file: File,
  onProgress?: (percent: number) => void,
): Promise<UploadResult> => {
  const form = new FormData()
  form.append('file', file)
  return http
    .post<UploadResult>('/upload', form, {
      timeout: 0, // 大文件不限时
      onUploadProgress: (e) => {
        if (onProgress && e.total) onProgress(Math.round((e.loaded / e.total) * 100))
      },
    })
    .then((r) => r.data)
}

export const startAnalyze = (fileId: string, name: string, options: AnalyzeOptions) =>
  http
    .post<{ job_id: string; video_url: string }>('/analyze', {
      file_id: fileId,
      name,
      options,
    })
    .then((r) => r.data)

export const probeLink = (url: string) =>
  http.post<ProbeResult>('/fetch/probe', { url }, { timeout: 90_000 }).then((r) => r.data)

export const startFetch = (url: string, options: AnalyzeOptions) =>
  http.post<{ job_id: string }>('/fetch', { url, options }, { timeout: 90_000 }).then((r) => r.data)

// ---------------------------------------------------------------- 任务

export const getJob = (id: string) => http.get<Job>(`/jobs/${id}`).then((r) => r.data)

export const listJobs = (limit = 30) =>
  http.get<JobSummary[]>('/jobs', { params: { limit } }).then((r) => r.data)

export const cancelJob = (id: string) =>
  http.post<{ ok: boolean; message: string }>(`/jobs/${id}/cancel`).then((r) => r.data)

export const deleteJob = (id: string) => http.delete<{ ok: boolean }>(`/jobs/${id}`).then((r) => r.data)

// ---------------------------------------------------------------- 提示词库

export const savePrompt = (payload: {
  job_id: string
  content: string
  title?: string
  format?: string
  tags?: string
  note?: string
}) => http.post<{ ok: boolean; id: string }>('/prompts', payload).then((r) => r.data)

export const listPrompts = (keyword?: string) =>
  http.get<Array<Record<string, unknown>>>('/prompts', { params: { keyword } }).then((r) => r.data)

export const deletePrompt = (id: string) =>
  http.delete<{ ok: boolean }>(`/prompts/${id}`).then((r) => r.data)

// ---------------------------------------------------------------- SSE

/**
 * 订阅任务进度。
 *
 * 用原生 EventSource（浏览器自动重连）。后端在任务结束时发 `event: close`，
 * 我们收到就主动断开，避免无意义的重连。
 */
export function subscribeJob(
  jobId: string,
  onEvent: (e: JobEvent) => void,
  onError?: (e: Event) => void,
): () => void {
  const es = new EventSource(`/api/jobs/${jobId}/events`)
  let closed = false

  const close = () => {
    if (!closed) {
      closed = true
      es.close()
    }
  }

  es.onmessage = (ev) => {
    try {
      const data = JSON.parse(ev.data) as JobEvent
      onEvent(data)
    } catch {
      /* 心跳或非 JSON 行，忽略 */
    }
  }

  es.addEventListener('close', () => {
    close()
    onEvent({ type: '__close__', ts: Date.now() })
  })

  es.onerror = (e) => {
    // 任务已结束时后端会关闭连接，这里的 error 属正常，不再上报
    if (closed) return
    onError?.(e)
  }

  return close
}
