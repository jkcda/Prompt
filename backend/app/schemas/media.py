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

    # ---- 频谱能量特征（ffmpeg 实测，不需要任何模型）----
    # 没有 ASR 时，这几个数字是唯一能拿到的音频信息。
    # ⚠️ 它们只能说明「能量分布在哪」，不能说明「是什么声音」。
    # 而且**不许把这些数字或它们的术语写进最终提示词** ——
    # 实测输出里出现过 "content unanalysed" / "voice band" 这类工具术语，
    # 视频模型拿到这种句子写不出任何东西。数字只用来支撑一句人类可读的描述。
    speech_band_db: float | None = None   # 300-3400Hz 相对全频段的能量（dB，越接近 0 越集中于人声频段）
    low_band_db: float | None = None      # <200Hz 相对全频段的能量（dB，越接近 0 低频越强）
    high_band_db: float | None = None     # >6kHz 相对全频段的能量（dB，掌声/镲片的线索）

    # ---- 节奏与音乐画像（纯 Python 从 PCM 算出）----
    bpm: float | None = None              # 估计的每分钟拍数
    has_beat: bool = False                # 有没有稳定节拍
    onset_rate: float = 0.0               # 每秒音头数，反映节奏密度
    transient_bursts: int = 0             # 密集高频瞬态簇个数（掌声/欢呼线索）
    dynamic_range_db: float | None = None  # 峰值 - 平均
    # 人类可读的音乐描述，由 audio_features.MusicProfile.describe() 生成。
    # 成文阶段可以直接用这句话，也可以改写，但**不许退回工具术语**。
    music_profile: str = ""

    # ---- 转写来源 ----
    language: str = ""                    # 转写检出的语言（保留原语言输出用）
    vocal_isolation: str = ""             # 人声是怎么提取的（Demucs / ffmpeg 带通 / 未处理）
    note: str = ""
