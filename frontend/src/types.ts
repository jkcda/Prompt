/** 与后端 `app/schemas` 一一对应的类型定义。 */

export type PromptFormat = 'h3' | 'h3-ref' | 'seedance' | 'generic'
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
  max_total_frames: number | null
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
  note: string
}

export interface ShotObservation {
  shot: string
  timecode: string
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
}

export interface JobResult {
  prompt: string
  observations: ShotObservation[]
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
