"""媒体与音频相关的数据模型。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class MediaInfo(BaseModel):
    """ffmpeg / ffprobe 探测出的媒体信息。"""

    path: str
    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    has_video: bool = False
    has_audio: bool = False
    video_codec: str = ""
    audio_codec: str = ""
    size_bytes: int = 0


class Shot(BaseModel):
    """一个镜头区间。由场景切换检测或均匀分镜产生。"""

    index: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class FrameRef(BaseModel):
    """一张送入模型的候选帧。"""

    shot_index: int
    time: float
    path: str
    role: Literal["head", "mid", "tail", "uniform"] = "mid"


class TranscriptSegment(BaseModel):
    """带时间戳的一句转写。"""

    start: float
    end: float
    text: str


class AudioReport(BaseModel):
    """音频维度报告。

    画面帧推不出台词、口型、BGM、音效，而目标提示词格式里这些是硬字段，
    所以这份报告是 Pass2 的必需输入。
    """

    has_audio: bool = False
    transcript: str = ""
    segments: list[TranscriptSegment] = Field(default_factory=list)
    mean_volume_db: float | None = None
    peak_volume_db: float | None = None
    silence_ratio: float | None = None
    loudness_points: list[float] = Field(default_factory=list)
    note: str = ""
