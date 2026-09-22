"""任务状态与进度事件（内存 + SQLite 双写）。

内存负责「活的任务」：进度事件、SSE 订阅、取消控制。
SQLite 负责「历史」：任务终态与结果落库，重启后仍能查。

每个任务维护一条事件历史 + 若干订阅队列：
  - 订阅者接入时先回放历史事件（避免刷新页面后进度丢失）；
  - 之后实时推送新事件；
  - 任务结束推送终态事件并关闭所有队列。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from collections import OrderedDict
from typing import Any

from ..schemas import Job, JobProgress, JobResult

log = logging.getLogger("jobs")

MAX_JOBS = 200


class JobStore:
    def __init__(self) -> None:
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._events: dict[str, list[dict]] = {}
        self._subs: dict[str, list[asyncio.Queue]] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._cancelled: set[str] = set()
        self._lock = asyncio.Lock()

    # -- 生命周期 ----------------------------------------------------------

    async def create(self, job: Job) -> Job:
        async with self._lock:
            job.id = job.id or uuid.uuid4().hex[:16]
            job.created_at = time.time()
            self._jobs[job.id] = job
            self._events[job.id] = []
            self._subs[job.id] = []
            while len(self._jobs) > MAX_JOBS:
                old_id, _ = self._jobs.popitem(last=False)
                self._events.pop(old_id, None)
                self._subs.pop(old_id, None)
                self._cancelled.discard(old_id)
        self._persist(job)
        return job

    def get(self, job_id: str) -> Job | None:
        """先查内存，未命中再查库（服务重启后仍能取到历史结果）。"""
        job = self._jobs.get(job_id)
        if job:
            return job
        try:
            from . import storage
            return storage.load_job(job_id)
        except Exception as exc:  # noqa: BLE001
            log.warning("从数据库读取任务 %s 失败: %s", job_id, exc)
            return None

    def list(self, limit: int = 50) -> list[Job]:
        items = list(self._jobs.values())
        items.sort(key=lambda j: j.created_at, reverse=True)
        return items[:limit]

    def attach_task(self, job_id: str, task: asyncio.Task) -> None:
        self._tasks[job_id] = task

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if not job or job.state in ("succeeded", "failed", "cancelled"):
            return False
        self._cancelled.add(job_id)
        task = self._tasks.get(job_id)
        if task and not task.done():
            task.cancel()
        return True

    def is_cancelled(self, job_id: str) -> bool:
        return job_id in self._cancelled

    def raise_if_cancelled(self, job_id: str) -> None:
        if job_id in self._cancelled:
            raise asyncio.CancelledError("任务已取消")

    # -- 进度与事件 --------------------------------------------------------

    async def progress(
        self,
        job_id: str,
        stage: str,
        percent: int,
        message: str = "",
        stage_label: str = "",
    ) -> None:
        job = self._jobs.get(job_id)
        if not job:
            return
        stage_changed = job.progress.stage != stage
        job.state = "running"
        job.progress = JobProgress(
            stage=stage,
            stage_label=stage_label or stage,
            percent=max(0, min(100, int(percent))),
            message=message,
        )
        # 只在阶段切换时落库，避免每个进度事件都写一次整任务
        if stage_changed:
            self._persist(job)
        await self.emit(job_id, {
            "type": "progress",
            "stage": stage,
            "stage_label": job.progress.stage_label,
            "percent": job.progress.percent,
            "message": message,
        })

    async def emit(self, job_id: str, event: dict) -> None:
        event = {**event, "ts": time.time()}
        history = self._events.get(job_id)
        if history is not None:
            history.append(event)
            if len(history) > 400:
                del history[:100]
        for q in list(self._subs.get(job_id, [])):
            with contextlib.suppress(asyncio.QueueFull):
                q.put_nowait(event)

    async def finish(self, job_id: str, result: JobResult | None, error: str = "") -> None:
        job = self._jobs.get(job_id)
        if not job:
            return
        if job_id in self._cancelled and not error:
            job.state = "cancelled"
        elif error:
            job.state = "failed"
            job.error = error
        else:
            job.state = "succeeded"
        job.result = result
        job.finished_at = time.time()
        job.progress = JobProgress(
            stage=job.state,
            stage_label={"succeeded": "完成", "failed": "失败", "cancelled": "已取消"}.get(job.state, job.state),
            percent=100 if job.state == "succeeded" else job.progress.percent,
            message=error or "反推完成",
        )
        self._persist(job)
        await self.emit(job_id, {
            "type": "done",
            "state": job.state,
            "error": error,
            "percent": job.progress.percent,
        })
        await self._close_subs(job_id)

    @staticmethod
    def _persist(job: Job) -> None:
        """落库。持久化失败不能影响主流程，只记日志。"""
        try:
            from . import storage
            storage.save_job(job)
        except Exception as exc:  # noqa: BLE001
            log.warning("任务 %s 落库失败: %s", job.id, exc)

    async def _close_subs(self, job_id: str) -> None:
        for q in list(self._subs.get(job_id, [])):
            with contextlib.suppress(asyncio.QueueFull):
                q.put_nowait({"type": "__close__", "ts": time.time()})

    # -- 订阅 --------------------------------------------------------------

    async def subscribe(self, job_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=500)
        async with self._lock:
            self._subs.setdefault(job_id, []).append(q)
            for event in self._events.get(job_id, []):
                try:
                    q.put_nowait(event)
                except asyncio.QueueFull:
                    break
            job = self._jobs.get(job_id)
            if job and job.state in ("succeeded", "failed", "cancelled"):
                q.put_nowait({"type": "__close__", "ts": time.time()})
        return q

    async def unsubscribe(self, job_id: str, q: asyncio.Queue) -> None:
        async with self._lock:
            subs = self._subs.get(job_id)
            if subs and q in subs:
                subs.remove(q)

    # -- 调试 --------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        by_state: dict[str, int] = {}
        for j in self._jobs.values():
            by_state[j.state] = by_state.get(j.state, 0) + 1
        return {
            "total": len(self._jobs),
            "by_state": by_state,
            "running_tasks": sum(1 for t in self._tasks.values() if not t.done()),
        }


store = JobStore()
