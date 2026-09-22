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
    """内存任务表 + 事件总线。

    这里刻意**不加 asyncio.Lock**：所有字典操作都是同步的（中间没有 await 点），
    在单线程事件循环里本身就是原子的。加了反而会在跨事件循环使用时抛
    "bound to a different event loop"——CLI 和测试需要在新循环里跑同一份 store。
    """

    def __init__(self) -> None:
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._events: dict[str, list[dict]] = {}
        self._subs: dict[str, list[asyncio.Queue]] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._cancelled: set[str] = set()

    # -- 生命周期 ----------------------------------------------------------

    async def create(self, job: Job) -> Job:
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
        # 判定顺序有讲究：显式取消优先，其次有错误就是失败，
        # 再次「没有结果」也算失败。
        if job_id in self._cancelled:
            job.state = "cancelled"
        elif error:
            job.state = "failed"
            job.error = error
        elif result is None:
            # 没产出结果就不算成功。
            # 踩过：服务关闭时 asyncio 任务被取消，CancelledError 分支调的是
            # finish(id, None, "")，按原来的逻辑会落到 succeeded ——
            # 界面上显示「完成」，点进去提示词是空的。这比直接报失败更难查，
            # 因为用户以为成功了。
            job.state = "failed"
            job.error = job.error or "任务被中断，未产出结果"
        else:
            job.state = "succeeded"
        job.result = result
        job.finished_at = time.time()
        job.progress = JobProgress(
            stage=job.state,
            stage_label={"succeeded": "完成", "failed": "失败", "cancelled": "已取消"}.get(job.state, job.state),
            percent=100 if job.state == "succeeded" else job.progress.percent,
            message=job.error or ("反推完成" if job.state == "succeeded" else "已取消"),
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
        subs = self._subs.get(job_id)
        if subs and q in subs:
            subs.remove(q)

    # -- 调试 --------------------------------------------------------------

    def events(self, job_id: str) -> list[dict]:
        """该任务已发出的事件（供测试与调试查看）。"""
        return list(self._events.get(job_id, []))

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
