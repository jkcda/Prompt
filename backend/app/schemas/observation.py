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


class SubjectEntry(BaseModel):
    """跨镜头复用的主体登记项。

    为什么要单独建模：
        `ShotObservation` 是逐镜头视角，同一个角色在第 1 镜和第 5 镜会被描述成
        两段互不相关的文字，模型无法判断「这是同一个人」。而 Ref2VA 的
        `subject_definitions`（把画风当独立主体锁、多参考图分工）和
        `retention_analysis`（逐主体写保留等级）恰恰需要跨镜头的主体身份，
        否则 Pass2 只能凭猜测编标签，参考标签会漂。

        所以 Pass1 除了逐镜头观察，还要额外交一份主体登记表：谁/什么、
        在第几镜出现、外观的哪些特征需要跨镜保持一致。
    """

    label: str = ""            # 建议标签，如 performer / rooftop / jacket / grade
    kind: str = ""             # person | environment | prop | wardrobe | style | other
    description: str = ""      # 外观、材质、颜色、可识别特征
    shots: list[str] = Field(default_factory=list)  # 出现的镜号（字符串）
    notes: str = ""            # 需要跨镜保持一致的要点 / 易漂移特征


class ChunkObservation(BaseModel):
    """一个分块的观察结果。长视频会被切成多个块并行分析。"""

    chunk_index: int
    start: float
    end: float
    shots: list[ShotObservation] = Field(default_factory=list)
    subjects: list[SubjectEntry] = Field(default_factory=list)
    global_notes: str = ""
    # 剪辑结构：模型自己判断的，不是我们切出来的。
    #
    # 为什么单独记：我们的「镜头」是 ffmpeg 场景检测切出来的，那只是**抽帧用的
    # 采样单位**，不是剪辑事实 —— 一镜到底的连续运镜经常被它切碎。
    # 用户反馈过「很多镜头其实是一镜到底的但是被切镜头了」。
    # 让模型从画面判断，成文阶段据此决定要不要输出切点标记。
    edit_structure: str = ""          # continuous | multi_shot
    cut_points: list[str] = Field(default_factory=list)   # 真实切点（MM:SS.mmm）
    continuity_notes: str = ""        # 判断依据
    raw: str = ""
