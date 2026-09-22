"""数据表模型。

设计取舍：结果类字段（media / audio / shots / observations / stats / options）存 JSON，
只把需要筛选排序的字段建成真实列。原因是 Pass1 的观察字段会随提示词迭代增删，
如果每次都改表结构，迁移成本远高于收益。
"""

from __future__ import annotations

import time

from sqlmodel import Field, SQLModel


def _now() -> float:
    return time.time()


class JobRecord(SQLModel, table=True):
    """一次反推任务及其结果。"""

    __tablename__ = "jobs"

    id: str = Field(primary_key=True)
    state: str = Field(default="pending", index=True)
    created_at: float = Field(default_factory=_now, index=True)
    finished_at: float | None = None

    source: str = Field(default="", index=True)      # upload | bilibili | douyin | url
    source_url: str = ""
    title: str = ""
    video_url: str = ""

    format: str = Field(default="h3", index=True)    # h3 | h3-ref | seedance | generic
    language: str = "en"
    error: str = ""

    frames_used: int = 0
    chunks: int = 0
    elapsed_sec: float = 0.0

    prompt: str = ""
    prompt_preview: str = Field(default="", index=True)

    # ---- JSON 列 ----
    options_json: str = "{}"
    progress_json: str = "{}"
    media_json: str = "{}"
    audio_json: str = "{}"
    shots_json: str = "[]"
    observations_json: str = "[]"
    subjects_json: str = "[]"
    frame_urls_json: str = "[]"
    stats_json: str = "{}"


class PromptRecord(SQLModel, table=True):
    """提示词库：把某次反推的结果收藏下来，可加标签备注。"""

    __tablename__ = "prompts"

    id: str = Field(primary_key=True)
    job_id: str = Field(default="", index=True)
    title: str = ""
    content: str = ""
    format: str = Field(default="h3", index=True)
    tags: str = ""                                    # 逗号分隔
    note: str = ""
    created_at: float = Field(default_factory=_now, index=True)
