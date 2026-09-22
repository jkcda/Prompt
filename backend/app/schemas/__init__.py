"""Pydantic 数据模型统一出口。

分层：
    media.py        媒体探测、镜头、帧、音频报告
    observation.py  Pass1 结构化镜头观察
    job.py          任务 / 进度 / 结果 / 反推选项
    api.py          HTTP 请求响应体
"""

from .api import (
    AnalyzeRequest,
    FetchProbeRequest,
    FetchRequest,
    FormatOption,
    HealthResponse,
    JobCreatedResponse,
    ProbeResult,
    PromptModeOption,
    UploadResponse,
)
from .job import (
    AnalyzeOptions,
    Job,
    JobProgress,
    JobResult,
    JobState,
    JobSummary,
    PromptFormat,
    PromptMode,
)
from .media import AudioReport, FrameRef, MediaInfo, Shot, TranscriptSegment
from .observation import ChunkObservation, ShotObservation, SubjectEntry

__all__ = [
    # media
    "MediaInfo", "Shot", "FrameRef", "TranscriptSegment", "AudioReport",
    # observation
    "ShotObservation", "ChunkObservation", "SubjectEntry",
    # job
    "AnalyzeOptions", "JobProgress", "JobResult", "Job", "JobSummary",
    "PromptFormat", "PromptMode", "JobState",
    # api
    "UploadResponse", "AnalyzeRequest", "FetchRequest", "FetchProbeRequest",
    "ProbeResult", "JobCreatedResponse", "FormatOption", "PromptModeOption",
    "HealthResponse",
]
