"""任务接口：列表、详情、SSE 进度、取消、删除，以及提示词库。"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from ..dependencies import StoreDep
from ..schemas import Job, JobSummary
from ..services import storage

log = logging.getLogger("api.jobs")
router = APIRouter(prefix="/api", tags=["jobs"])


@router.get("/jobs", response_model=list[JobSummary], summary="任务历史列表")
async def list_jobs(
    jobs: StoreDep,
    limit: int = Query(30, ge=1, le=200),
    offset: int = Query(0, ge=0),
    state: str | None = None,
    source: str | None = None,
    keyword: str | None = None,
) -> list[JobSummary]:
    """走数据库查历史（含服务重启前的记录），内存里的活跃任务单独合并进来。"""
    items = await asyncio.to_thread(
        storage.list_jobs, limit, offset, state, source, keyword
    )
    seen = {i.id for i in items}
    if offset == 0:
        for live in jobs.list(limit):
            if live.id not in seen and (not state or live.state == state):
                items.insert(0, JobSummary(
                    id=live.id,
                    state=live.state,  # type: ignore[arg-type]
                    created_at=live.created_at,
                    finished_at=live.finished_at,
                    source=live.source,
                    source_url=live.source_url,
                    title=live.title,
                    video_url=live.video_url,
                    format=live.options.format,
                    prompt_preview=(live.result.prompt[:280] if live.result else ""),
                    frames_used=live.result.frames_used if live.result else 0,
                    elapsed_sec=float(live.result.stats.get("elapsed_sec") or 0) if live.result else 0.0,
                    error=live.error,
                ))
    items.sort(key=lambda x: x.created_at, reverse=True)
    return items[:limit]


@router.get("/jobs/{job_id}", response_model=Job, summary="任务详情（含提示词与观察结果）")
async def get_job(job_id: str, jobs: StoreDep) -> Job:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "任务不存在")
    return job


@router.post("/jobs/{job_id}/cancel", summary="取消任务")
async def cancel_job(job_id: str, jobs: StoreDep) -> dict[str, Any]:
    ok = jobs.cancel(job_id)
    return {"ok": ok, "message": "已发送取消信号" if ok else "任务不在可取消状态"}


@router.delete("/jobs/{job_id}", summary="删除任务记录")
async def delete_job(job_id: str, jobs: StoreDep) -> dict[str, Any]:
    live = jobs.get(job_id)
    if live and live.state == "running":
        raise HTTPException(409, "任务正在运行，请先取消")
    ok = await asyncio.to_thread(storage.delete_job, job_id)
    return {"ok": ok}


@router.get("/jobs/{job_id}/events", summary="SSE 实时进度")
async def job_events(job_id: str, request: Request, jobs: StoreDep) -> StreamingResponse:
    if not jobs.get(job_id):
        raise HTTPException(404, "任务不存在")

    async def _stream():
        queue = await jobs.subscribe(job_id)
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=20.0)
                except asyncio.TimeoutError:
                    # 心跳，防止中间代理掐掉空闲连接
                    yield ": keep-alive\n\n"
                    continue
                if event.get("type") == "__close__":
                    yield f"event: close\ndata: {json.dumps({'ts': event['ts']})}\n\n"
                    break
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        finally:
            await jobs.unsubscribe(job_id, queue)

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# 提示词库
# ---------------------------------------------------------------------------

@router.post("/prompts", summary="收藏一条提示词")
async def save_prompt(payload: dict[str, Any]) -> dict[str, Any]:
    content = str(payload.get("content") or "").strip()
    if not content:
        raise HTTPException(400, "content 不能为空")
    pid = await asyncio.to_thread(
        storage.save_prompt,
        str(payload.get("job_id") or ""),
        content,
        str(payload.get("title") or ""),
        str(payload.get("format") or "h3"),
        str(payload.get("tags") or ""),
        str(payload.get("note") or ""),
    )
    return {"ok": True, "id": pid}


@router.get("/prompts", summary="提示词库列表")
async def list_prompts(limit: int = Query(50, ge=1, le=500), keyword: str | None = None) -> list[dict]:
    return await asyncio.to_thread(storage.list_prompts, limit, keyword)


@router.delete("/prompts/{prompt_id}", summary="删除收藏")
async def delete_prompt(prompt_id: str) -> dict[str, Any]:
    ok = await asyncio.to_thread(storage.delete_prompt, prompt_id)
    return {"ok": ok}
