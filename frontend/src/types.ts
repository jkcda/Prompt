/** 与后端 `app/schemas` 一一对应的类型定义。 */

export type PromptFormat = 'h3' | 'h3-ref' | 'seedance' | 'generic'
export type PromptMode = 'h3' | 'seedance' | 'generic'
export type JobState = 'pending' | 'running' | 'succeeded' | 'failed' | 'cancelled'
export type JobSource = 'upload' | 'bilibili' | 'douyin' | 'url'
export type FrameRole = 'head' | 'mid' | 'tail' | 'uniform'

export interface UploadResult {
  file_id: string
  name: string
  size: number
  video_url: string
  duration: number
  width: number
  height: number
  fps: number
  has_audio: boolean
}

export interface AnalyzeOptions {
  format: PromptFormat
  language: 'zh' | 'en'
  enable_asr: boolean
  enable_scene_split: boolean
  /** 以下留空（null）表示用服务端 .env 里的默认值 */
  max_total_frames: number | null
  frame_interval_seconds: number | null
  prompt_word_limit: number | null
  /** 只反推这段区间（秒，相对原片）。两个都为 null 表示整片。 */
  trim_start: number | null
  trim_end: number | null
  /** 用户自己写的画面说明，喂给观察阶段和成文阶段。 */
  content_hint: string
  extra_instruction: string
  target_duration: number | null
}

export interface MediaInfo {
  path: string
  duration: number
  width: number
  height: number
  fps: number
  has_video: boolean
  has_audio: boolean
  video_codec: string
  audio_codec: string
  size_bytes: number
}

export interface Shot {
  index: number
  start: number
  end: number
}

export interface TranscriptSegment {
  start: number
  end: number
  text: string
}

export interface AudioReport {
  has_audio: boolean
  transcript: string
  segments: TranscriptSegment[]
  mean_volume_db: number | null
  peak_volume_db: number | null
  silence_ratio: number | null
  loudness_points: number[]
  // 频段能量（dB，相对全频段）。**仅诊断用** —— 这些数字不会写进提示词，
  // 它们只在服务端支撑 music_profile 那句人类可读的描述。
  speech_band_db: number | null
  low_band_db: number | null
  high_band_db: number | null
  // 节奏与音乐画像（纯 Python 从 PCM 算出，不需要模型）
  bpm: number | null
  has_beat: boolean
  onset_rate: number
  transient_bursts: number
  dynamic_range_db: number | null
  /** 人类可读的音乐描述，成文阶段会直接用它 */
  music_profile: string
  /** 转写检出的语言，歌词要保持原语言 */
  language: string
  /** 人声是怎么提取的（Demucs / ffmpeg 带通 / 未处理） */
  vocal_isolation: string
  note: string
}

export interface ShotObservation {
  shot: string
  timecode: string
  /** 该镜头在源视频里的起止毫秒。后端会补全并对齐，末镜的 end_ms 等于总时长。 */
  start_ms: number
  end_ms: number
  shot_size: string
  camera: string
  subject: string
  action: string
  setting: string
  lighting: string
  color: string
  motion_energy: string
  on_screen_text: string
  dialogue: string
  sfx: string
  transition: string
  confidence: number
  /** 这一条由代码把相邻的同一机位条目合并而来 */
  is_continuous: boolean
}

/** Pass1 的跨镜头主体登记项，Ref2VA 的参考标签由它推导。 */
export interface SubjectEntry {
  label: string
  kind: string
  description: string
  shots: string[]
  notes: string
}

export interface JobResult {
  prompt: string
  observations: ShotObservation[]
  subjects: SubjectEntry[]
  media: MediaInfo | null
  audio: AudioReport | null
  shots: Shot[]
  frames_used: number
  frame_urls: string[]
  chunks: number
  stats: Record<string, unknown>
}

export interface JobProgress {
  stage: string
  stage_label: string
  percent: number
  message: string
}

export interface Job {
  id: string
  state: JobState
  created_at: number
  finished_at: number | null
  source: string
  source_url: string
  title: string
  video_url: string
  options: AnalyzeOptions
  progress: JobProgress
  result: JobResult | null
  error: string
}

export interface JobSummary {
  id: string
  state: JobState
  created_at: number
  finished_at: number | null
  source: string
  source_url: string
  title: string
  video_url: string
  format: string
  prompt_preview: string
  frames_used: number
  elapsed_sec: number
  error: string
}

export interface ProbeResult {
  platform: string
  title: string
  duration: number
  thumbnail: string
  uploader: string
  direct_url: string
  note: string
}

export interface FormatOption {
  value: PromptFormat
  label: string
  description: string
  mode: PromptMode
  default: boolean
}

/**
 * 一种反推模式及其变体。
 *
 * 对外只有两种模式：H3 与 Seedance。`generic` 是工具无关的兜底，
 * `primary: false`，界面上折进「其他格式」，不占主选择位。
 */
export interface PromptModeOption {
  value: PromptMode
  label: string
  description: string
  primary: boolean
  variants: FormatOption[]
}

export interface HealthInfo {
  ok: boolean
  ffmpeg: string | null
  ffprobe: string | null
  vlm_configured: boolean
  vlm_model: string
  asr_configured: boolean
  ytdlp: boolean
  jobs: Record<string, unknown>
  db: string
}

/** 模型自检结果。`vision` 才是关键 —— 它代表模型真的读到了图片内容。 */
export interface ModelTestResult {
  configured: boolean
  base_url: string
  model: string
  ok: boolean
  /** 真的识别出了自检图片（纯红图答出了 red） */
  vision: boolean
  /** 无法判定：有输出但被 max_tokens 截断（推理模型思考过程很长） */
  inconclusive: boolean
  message: string
}

export interface ModelListResult {
  ok: boolean
  models: string[]
  message: string
}

/** `GET /api/settings` 的响应。只包含可热更新的项。 */
export interface SettingsInfo {
  vlm: {
    configured: boolean
    base_url: string
    model: string
    concurrency: number
    audio_input: boolean
    api_key_masked: string
  }
  asr: {
    configured: boolean
    base_url: string
    model: string
    api_key_masked: string
  }
  frames: {
    max_total_frames: number
    max_frames_per_shot: number
    frame_interval_seconds: number
    long_shot_seconds: number
    long_edge: number
    jpeg_quality: number
  }
  prompt: { word_limit: number }
  scene: { threshold: number; min_shot_seconds: number }
  chunk: { threshold_seconds: number; seconds: number; overlap_seconds: number }
  upload: { max_mb: number }
  database: { url: string; path: string }
  ffmpeg: { path: string | null; ffprobe: string | null }
}

/** SSE 事件（后端 `store.emit` 发出的载荷）。 */
export interface JobEvent {
  type: 'progress' | 'chunk' | 'fetched' | 'video_ready' | 'done' | '__close__'
  ts: number
  stage?: string
  stage_label?: string
  percent?: number
  message?: string
  state?: JobState
  error?: string
  index?: number
  total?: number
  shots?: number
  title?: string
  uploader?: string
  duration?: number
  platform?: string
  video_url?: string
}
