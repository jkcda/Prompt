"""Pass1（结构化镜头观察）的输出模型。

字段设计对应「反推提示词」真正需要的维度，而不是泛泛的画面描述：
  - `camera`      : 运镜方向与幅度，直接决定生成视频的镜头语言
  - `action`      : 必须含物理传导细节，否则生成结果会僵硬如静帧
  - `dialogue`    : 台词/歌词原文，Pass2 要用它写 `<d>` 段落
  - `sfx`         : 非语音声，Pass2 要汇总进 overall_soundscape
  - `transition`  : 镜头衔接方式，决定切点怎么写
  - `confidence`  : 低置信度让 Pass2 知道哪些细节该含糊处理
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ShotObservation(BaseModel):
    shot: str = ""
    timecode: str = ""
    shot_size: str = ""       # 景别：特写 / 近景 / 中景 / 全景 / 远景
    camera: str = ""          # 运镜：固定 / 推 / 拉 / 摇 / 移 / 跟 / 手持 / 环绕 / 升降
    subject: str = ""         # 主体与外观、服装、关键道具
    action: str = ""          # 动作 + 物理传导细节
    setting: str = ""         # 环境与场景
    lighting: str = ""        # 光线来源、方向、质感
    color: str = ""           # 色调与影调
    motion_energy: str = ""   # 运动强度与节奏
    on_screen_text: str = ""  # 画面内文字（原样抄录）
    dialogue: str = ""        # 该镜头内的台词/歌词（原文）
    sfx: str = ""             # 非语音声
    transition: str = ""      # 与下一镜的衔接
    confidence: float = 0.0


class ChunkObservation(BaseModel):
    """一个分块的观察结果。长视频会被切成多个块并行分析。"""

    chunk_index: int
    start: float
    end: float
    shots: list[ShotObservation] = Field(default_factory=list)
    global_notes: str = ""
    raw: str = ""
