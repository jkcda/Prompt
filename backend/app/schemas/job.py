"""任务、进度、结果模型。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from .media import AudioReport, MediaInfo, Shot
from .observation import ShotObservation, SubjectEntry

PromptFormat = Literal["h3", "h3-ref", "seedance", "generic"]
PromptMode = Literal["h3", "seedance", "generic"]
JobState = Literal["pending", "running", "succeeded", "failed", "cancelled"]
JobSource = Literal["upload", "bilibili", "douyin", "url"]


class AnalyzeOptions(BaseModel):
    """一次反推的可调参数。前端「高级选项」面板对应这里。

    `format` 是唯一的格式真源，`mode` 由它推导（见 `templates.MODE_OF_FORMAT`），
    不额外传参——否则会出现 mode=h3 而 format=seedance 这种自相矛盾的组合。
    """

    format: PromptFormat = "h3"
    language: Literal["zh", "en"] = "en"
    enable_asr: bool = True
    enable_scene_split: bool = True
    # 以下留空表示用服务端 .env 里的默认值
    max_total_frames: int | None = None
    frame_interval_seconds: float | None = None
    prompt_word_limit: int | None = None
    # 只反推这段区间（秒，相对原片）。留空表示整片。
    # 用户可以先看片再自由框选，避免把无关的前后内容也写进提示词。
    trim_start: float | None = None
    trim_end: float | None = None
    extra_instruction: str = ""
    target_duration: float | None = None

    @model_validator(mode="after")
    def _check_trim(self) -> AnalyzeOptions:
        """区间校验：起止都要有（或都没有），且 end > start。"""
        a, b = self.trim_start, self.trim_end
        if (a is None) != (b is None):
            raise ValueError("trim_start 和 trim_end 必须同时提供或同时留空")
        if a is not None and b is not None:
            if a < 0:
                raise ValueError("trim_start 不能为负")
            if b <= a:
                raise ValueError(f"trim_end（{b}）必须大于 trim_start（{a}）")
        return self


class JobProgress(BaseModel):
    stage: str = "pending"
    stage_label: str = ""
    percent: int = 0
    message: str = ""


class JobResult(BaseModel):
    prompt: str = ""
    observations: list[ShotObservation] = Field(default_factory=list)
    subjects: list[SubjectEntry] = Field(default_factory=list)
    media: MediaInfo | None = None
    audio: AudioReport | None = None
    shots: list[Shot] = Field(default_factory=list)
    frames_used: int = 0
    frame_urls: list[str] = Field(default_factory=list)
    chunks: int = 0
    stats: dict[str, Any] = Field(default_factory=dict)


class Job(BaseModel):
    id: str
    state: JobState = "pending"
    created_at: float = 0.0
    finished_at: float | None = None
    source: str = ""
    source_url: str = ""
    title: str = ""
    video_url: str = ""
    options: AnalyzeOptions = Field(default_factory=AnalyzeOptions)
    progress: JobProgress = Field(default_factory=JobProgress)
    result: JobResult | None = None
    error: str = ""


class JobSummary(BaseModel):
    """历史列表用的轻量视图（不含 observations / frame_urls，避免列表响应过大）。"""

    id: str
    state: JobState
    created_at: float
    finished_at: float | None = None
    source: str = ""
    source_url: str = ""
    title: str = ""
    video_url: str = ""
    format: str = ""
    prompt_preview: str = ""
    frames_used: int = 0
    elapsed_sec: float = 0.0
    error: str = ""
