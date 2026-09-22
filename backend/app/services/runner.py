"""任务启动器：把「创建任务 → 后台跑管线 → 落库」这段编排从路由里抽出来。

路由只负责校验请求和返回 job_id，不关心任务怎么跑。
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from pathlib import Path

from ..schemas import AnalyzeOptions, Job
from . import downloader
from .jobs import store
from .pipeline import run_pipeline

log = logging.getLogger("runner")


def new_job_id() -> str:
    return uuid.uuid4().hex[:16]


def parse_options(payload: dict) -> AnalyzeOptions:
    """从请求体里宽松解析反推选项（未知键忽略，不报错）。"""
    raw = payload.get("options") or payload
    try:
        return AnalyzeOptions(**{
            k: v for k, v in raw.items() if k in AnalyzeOptions.model_fields
        })
    except Exception:  # noqa: BLE001
        return AnalyzeOptions()


async def start_upload_job(
    path: Path,
    video_url: str,
    title: str,
    options: AnalyzeOptions,
) -> Job:
    """对已上传的本地文件发起反推。"""
    job = await store.create(Job(
        id=new_job_id(),
        source="upload",
        title=title,
        video_url=video_url,
        options=options,
    ))
    _spawn(job, path, source_url="")
    return job


async def start_fetch_job(url: str, options: AnalyzeOptions) -> Job:
    """对 B站/抖音链接发起「抓取 + 反推」。"""
    job = await store.create(Job(
        id=new_job_id(),
        source=downloader.detect_platform(url),
        source_url=url,
        options=options,
    ))
    _spawn(job, None, source_url=url)
    return job


def _spawn(job: Job, path: Path | None, source_url: str) -> None:
    async def _runner() -> None:
        try:
            video_path = path

            # ---- 抓取阶段（仅链接任务）----
            if video_path is None:
                await store.progress(job.id, "fetch", 2, "下载视频", "抓取视频")
                got, info, err = await asyncio.to_thread(
                    downloader.download, source_url, job.id
                )
                if not got:
                    await store.finish(job.id, None, err or "视频抓取失败")
                    return

                video_path = got
                job.title = info.title or job.title
                await store.emit(job.id, {
                    "type": "fetched",
                    "title": info.title,
                    "uploader": info.uploader,
                    "duration": info.duration,
                    "platform": info.platform,
                })

                # 挪到 uploads 目录，让前端能直接播放
                target = _uploads_dir() / f"fetch_{job.id}{video_path.suffix}"
                try:
                    os.replace(video_path, target)
                    video_path = target
                except OSError as exc:
                    log.warning("下载件移动失败，就地使用: %s", exc)

                job.video_url = f"/api/media/upload/{video_path.name}"
                await store.emit(job.id, {"type": "video_ready", "video_url": job.video_url})

            store.raise_if_cancelled(job.id)

            # ---- 反推阶段 ----
            result = await run_pipeline(job, video_path)
            await store.finish(job.id, result)

        except asyncio.CancelledError:
            await store.finish(job.id, None, "")
        except Exception as exc:  # noqa: BLE001
            log.exception("任务 %s 失败", job.id)
            await store.finish(job.id, None, str(exc))

    task = asyncio.create_task(_runner())
    store.attach_task(job.id, task)


def _uploads_dir() -> Path:
    from ..core.config import UPLOAD_DIR
    return UPLOAD_DIR
