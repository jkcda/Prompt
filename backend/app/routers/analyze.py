"""反推接口：本地文件反推、链接抓取反推、链接预探测。"""

from __future__ import annotations

import asyncio
import logging
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException

from ..core.config import UPLOAD_DIR
from ..schemas import (
    AnalyzeRequest,
    FetchProbeRequest,
    FetchRequest,
    JobCreatedResponse,
    ProbeResult,
    UploadResponse,
)
from ..services import downloader
from ..services import ffmpeg as ff
from ..services.runner import parse_options, start_fetch_job, start_upload_job
from .upload import _make_file_id

log = logging.getLogger("api.analyze")
router = APIRouter(prefix="/api", tags=["analyze"])


@router.post("/analyze", response_model=JobCreatedResponse, summary="对已上传视频发起反推")
async def analyze(req: AnalyzeRequest) -> JobCreatedResponse:
    path = UPLOAD_DIR / Path(req.file_id).name
    if not path.is_file():
        raise HTTPException(404, "文件不存在，请重新上传")

    job = await start_upload_job(
        path=path,
        video_url=f"/api/media/upload/{path.name}",
        title=req.name or path.name,
        options=req.options,
    )
    log.info("反推任务已创建：%s（%s）", job.id, job.options.format)
    return JobCreatedResponse(job_id=job.id, video_url=job.video_url)


@router.post("/fetch/probe", response_model=ProbeResult, summary="探测视频链接")
async def fetch_probe(req: FetchProbeRequest) -> ProbeResult:
    """只解析链接信息不下载，让前端先确认「是不是这个视频」。"""
    url = downloader.extract_url(req.url) or req.url
    if not url:
        raise HTTPException(400, "请提供视频链接或分享文案")
    return await asyncio.to_thread(downloader.probe, url)


@router.post(
    "/fetch/download",
    response_model=UploadResponse,
    summary="抓取链接并只下载（供用户先框选片段）",
)
async def fetch_download(req: FetchRequest) -> UploadResponse:
    """下载链接里的视频并落盘，返回和上传一样的信息。

    为什么要和 `/fetch` 分开：用户需要**先拿到视频、看过再框选**要反推的片段。
    `/fetch` 是「下载完直接反推」，中间没有让用户介入的机会。
    拆开之后前端可以先放预览、让用户拖出区间，再带着 trim 参数调 `/analyze`。
    """
    url = downloader.extract_url(req.url) or req.url
    if not url:
        raise HTTPException(400, "请提供视频链接或分享文案")

    # 借用任务目录做下载落地，成功后挪进 uploads（和 runner 的抓取流程一致）
    tag = f"dl-{uuid.uuid4().hex[:8]}"
    path, info, why = await asyncio.to_thread(downloader.download, url, tag)
    if path is None:
        raise HTTPException(400, why or "视频下载失败")

    file_id = _make_file_id(path.stem, path.suffix or ".mp4")
    dst = UPLOAD_DIR / file_id
    try:
        shutil.move(str(path), str(dst))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"保存下载文件失败：{exc}") from exc

    probed = await asyncio.to_thread(ff.probe, dst)
    size = dst.stat().st_size
    log.info("抓取下载完成：%s（%.1fMB，%.1fs）", file_id, size / 1048576, probed.duration)
    return UploadResponse(
        file_id=file_id,
        name=info.title or path.name,
        size=size,
        video_url=f"/api/media/upload/{file_id}",
        duration=probed.duration,
        width=probed.width,
        height=probed.height,
        fps=probed.fps,
        has_audio=probed.has_audio,
    )


@router.post("/fetch", response_model=JobCreatedResponse, summary="抓取链接并反推")
async def fetch_and_analyze(req: FetchRequest) -> JobCreatedResponse:
    url = downloader.extract_url(req.url) or req.url
    if not url:
        raise HTTPException(400, "请提供视频链接或分享文案")

    job = await start_fetch_job(url, req.options)
    log.info("抓取任务已创建：%s（%s，%s）", job.id, job.source, url[:80])
    return JobCreatedResponse(job_id=job.id)


@router.post("/probe", response_model=ProbeResult, summary="探测本地/远端媒体（调试用）")
async def probe_path(payload: dict) -> ProbeResult:
    path = str(payload.get("path") or "")
    if not path:
        raise HTTPException(400, "缺少 path")
    p = Path(path)
    if not p.is_file():
        raise HTTPException(404, "文件不存在")
    from ..services import ffmpeg as ff
    info = await asyncio.to_thread(ff.probe, p)
    return ProbeResult(
        platform="local",
        title=p.name,
        duration=info.duration,
        note=f"{info.width}x{info.height} @ {info.fps:.2f}fps, audio={info.has_audio}",
    )


# 保留旧路径的兼容入口，避免前端早期版本 404
@router.post("/fetch-legacy", include_in_schema=False)
async def fetch_legacy(payload: dict) -> dict:
    options = parse_options(payload)
    url = downloader.extract_url(str(payload.get("url") or "")) or str(payload.get("url") or "")
    job = await start_fetch_job(url, options)
    return {"job_id": job.id}
