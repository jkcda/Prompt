"""HTTP 请求/响应模型。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .job import AnalyzeOptions, PromptFormat


class UploadResponse(BaseModel):
    file_id: str
    name: str
    size: int
    video_url: str
    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    has_audio: bool = False


class AnalyzeRequest(BaseModel):
    file_id: str
    name: str = ""
    options: AnalyzeOptions = Field(default_factory=AnalyzeOptions)


class FetchRequest(BaseModel):
    """从 B站/抖音链接抓取并反推。`url` 允许直接贴分享文案。"""

    url: str
    options: AnalyzeOptions = Field(default_factory=AnalyzeOptions)


class FetchProbeRequest(BaseModel):
    url: str


class ProbeResult(BaseModel):
    platform: str = ""
    title: str = ""
    duration: float = 0.0
    thumbnail: str = ""
    uploader: str = ""
    direct_url: str = ""
    note: str = ""


class JobCreatedResponse(BaseModel):
    job_id: str
    video_url: str = ""


class FormatOption(BaseModel):
    value: PromptFormat
    label: str
    description: str = ""


class HealthResponse(BaseModel):
    ok: bool = True
    ffmpeg: str | None = None
    ffprobe: str | None = None
    vlm_configured: bool = False
    vlm_model: str = ""
    asr_configured: bool = False
    ytdlp: bool = False
    jobs: dict[str, Any] = Field(default_factory=dict)
    db: str = ""
