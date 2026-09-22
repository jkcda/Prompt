"""持久化读写：把 Job 模型与 SQLite 表互相转换。

分层约定：routers 不直接碰 Session，只调这里的函数。
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

from sqlmodel import Session, delete, select

from ..core.db import get_engine
from ..models.job import JobRecord, PromptRecord
from ..schemas import (
    AnalyzeOptions,
    AudioReport,
    Job,
    JobProgress,
    JobResult,
    JobSummary,
    MediaInfo,
    Shot,
    ShotObservation,
    SubjectEntry,
)

log = logging.getLogger("storage")


# ---------------------------------------------------------------------------
# 序列化辅助
# ---------------------------------------------------------------------------

def _dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return "{}"


def _loads(raw: str, default: Any) -> Any:
    if not raw:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default


# ---------------------------------------------------------------------------
# Job
# ---------------------------------------------------------------------------

def save_job(job: Job) -> None:
    """整任务落库（upsert）。"""
    result = job.result
    with Session(get_engine()) as session:
        record = session.get(JobRecord, job.id) or JobRecord(id=job.id)

        record.state = job.state
        record.created_at = job.created_at or time.time()
        record.finished_at = job.finished_at
        record.source = job.source
        record.source_url = job.source_url
        record.title = job.title
        record.video_url = job.video_url
        record.format = job.options.format
        record.language = job.options.language
        record.error = job.error
        record.options_json = _dumps(job.options.model_dump())
        record.progress_json = _dumps(job.progress.model_dump())

        if result:
            record.frames_used = result.frames_used
            record.chunks = result.chunks
            record.elapsed_sec = float(result.stats.get("elapsed_sec") or 0.0)
            record.prompt = result.prompt
            record.prompt_preview = (result.prompt or "")[:280]
            record.media_json = _dumps(result.media.model_dump() if result.media else {})
            record.audio_json = _dumps(result.audio.model_dump() if result.audio else {})
            record.shots_json = _dumps([s.model_dump() for s in result.shots])
            record.observations_json = _dumps([o.model_dump() for o in result.observations])
            record.subjects_json = _dumps([s.model_dump() for s in result.subjects])
            record.frame_urls_json = _dumps(result.frame_urls)
            record.stats_json = _dumps(result.stats)

        session.add(record)
        session.commit()


def load_job(job_id: str) -> Job | None:
    with Session(get_engine()) as session:
        record = session.get(JobRecord, job_id)
        if not record:
            return None
        return _to_job(record)


def list_jobs(
    limit: int = 30,
    offset: int = 0,
    state: str | None = None,
    source: str | None = None,
    keyword: str | None = None,
) -> list[JobSummary]:
    with Session(get_engine()) as session:
        stmt = select(JobRecord)
        if state:
            stmt = stmt.where(JobRecord.state == state)
        if source:
            stmt = stmt.where(JobRecord.source == source)
        if keyword:
            like = f"%{keyword}%"
            stmt = stmt.where(
                (JobRecord.title.like(like)) | (JobRecord.prompt_preview.like(like))
            )
        stmt = stmt.order_by(JobRecord.created_at.desc()).offset(offset).limit(limit)
        return [_to_summary(r) for r in session.exec(stmt).all()]


def count_jobs() -> int:
    with Session(get_engine()) as session:
        return len(session.exec(select(JobRecord)).all())


def delete_job(job_id: str) -> bool:
    with Session(get_engine()) as session:
        record = session.get(JobRecord, job_id)
        if not record:
            return False
        session.delete(record)
        session.commit()
        return True


def mark_interrupted() -> int:
    """启动时对账，修掉两类「僵尸」记录。

    1. 还挂在 `running` / `pending` 的 —— 任务状态只在内存里活着，终态才落库。
       服务被 kill、崩溃、断电时，最后那个任务会永远停在 running，
       界面上看起来一直在跑，其实进程早没了。
    2. `succeeded` 但 prompt 是空的 —— 这是「假成功」。服务关闭时 asyncio
       任务被取消，旧逻辑会把「无结果」判成成功，界面上显示「完成」，
       点进去提示词是空的。比直接报失败更难查，因为用户以为成功了。

    第 2 类按理不该再产生（`JobStore.finish` 已修），这里是对历史数据的兜底。
    """
    fixed = 0
    with Session(get_engine()) as session:
        stale = list(session.exec(
            select(JobRecord).where(JobRecord.state.in_(("running", "pending")))  # type: ignore[attr-defined]
        ))
        for record in stale:
            record.state = "failed"
            record.error = "服务重启，任务被中断"
            record.finished_at = record.finished_at or time.time()
            session.add(record)

        ghosts = [
            r for r in session.exec(
                select(JobRecord).where(JobRecord.state == "succeeded")  # type: ignore[arg-type]
            )
            if not (r.prompt or "").strip()
        ]
        for record in ghosts:
            record.state = "failed"
            record.error = "任务被中断，未产出结果（历史数据修复）"
            record.finished_at = record.finished_at or time.time()
            session.add(record)

        fixed = len(stale) + len(ghosts)
        if fixed:
            session.commit()
            log.warning("启动对账：收掉 %d 个被中断的任务，修掉 %d 条假成功记录",
                        len(stale), len(ghosts))
        return fixed


def cleanup_old(days: int) -> int:
    """删除超过 N 天的历史任务。days<=0 时不清理。"""
    if days <= 0:
        return 0
    cutoff = time.time() - days * 86400
    with Session(get_engine()) as session:
        stmt = delete(JobRecord).where(JobRecord.created_at < cutoff)
        result = session.exec(stmt)
        session.commit()
        removed = result.rowcount or 0
        if removed:
            log.info("清理 %d 条超过 %d 天的历史任务", removed, days)
        return removed


# ---------------------------------------------------------------------------
# 提示词库
# ---------------------------------------------------------------------------

def save_prompt(
    job_id: str,
    content: str,
    title: str = "",
    fmt: str = "h3",
    tags: str = "",
    note: str = "",
) -> str:
    pid = uuid.uuid4().hex[:16]
    with Session(get_engine()) as session:
        session.add(PromptRecord(
            id=pid, job_id=job_id, title=title, content=content,
            format=fmt, tags=tags, note=note,
        ))
        session.commit()
    return pid


def list_prompts(limit: int = 50, keyword: str | None = None) -> list[dict]:
    with Session(get_engine()) as session:
        stmt = select(PromptRecord)
        if keyword:
            like = f"%{keyword}%"
            stmt = stmt.where(
                (PromptRecord.title.like(like))
                | (PromptRecord.content.like(like))
                | (PromptRecord.tags.like(like))
            )
        stmt = stmt.order_by(PromptRecord.created_at.desc()).limit(limit)
        return [
            {
                "id": r.id, "job_id": r.job_id, "title": r.title,
                "content": r.content, "format": r.format,
                "tags": r.tags, "note": r.note, "created_at": r.created_at,
            }
            for r in session.exec(stmt).all()
        ]


def delete_prompt(prompt_id: str) -> bool:
    with Session(get_engine()) as session:
        record = session.get(PromptRecord, prompt_id)
        if not record:
            return False
        session.delete(record)
        session.commit()
        return True


# ---------------------------------------------------------------------------
# 行 → 模型
# ---------------------------------------------------------------------------

def _to_job(r: JobRecord) -> Job:
    result: JobResult | None = None
    if r.prompt or r.observations_json != "[]":
        media_raw = _loads(r.media_json, {})
        audio_raw = _loads(r.audio_json, {})
        result = JobResult(
            prompt=r.prompt,
            observations=[ShotObservation(**o) for o in _loads(r.observations_json, [])],
            subjects=[SubjectEntry(**s) for s in _loads(getattr(r, "subjects_json", None) or "[]", [])],
            media=MediaInfo(**media_raw) if media_raw else None,
            audio=AudioReport(**audio_raw) if audio_raw else None,
            shots=[Shot(**s) for s in _loads(r.shots_json, [])],
            frames_used=r.frames_used,
            frame_urls=_loads(r.frame_urls_json, []),
            chunks=r.chunks,
            stats=_loads(r.stats_json, {}),
        )

    options_raw = _loads(r.options_json, {})
    return Job(
        id=r.id,
        state=r.state,  # type: ignore[arg-type]
        created_at=r.created_at,
        finished_at=r.finished_at,
        source=r.source,
        source_url=r.source_url,
        title=r.title,
        video_url=r.video_url,
        options=AnalyzeOptions(**options_raw) if options_raw else AnalyzeOptions(),
        progress=JobProgress(**_loads(r.progress_json, {})),
        result=result,
        error=r.error,
    )


def _to_summary(r: JobRecord) -> JobSummary:
    return JobSummary(
        id=r.id,
        state=r.state,  # type: ignore[arg-type]
        created_at=r.created_at,
        finished_at=r.finished_at,
        source=r.source,
        source_url=r.source_url,
        title=r.title,
        video_url=r.video_url,
        format=r.format,
        prompt_preview=r.prompt_preview,
        frames_used=r.frames_used,
        elapsed_sec=r.elapsed_sec,
        error=r.error,
    )
