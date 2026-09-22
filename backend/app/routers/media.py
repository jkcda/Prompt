"""媒体接口：视频流（Range）、抽帧图片、缩略图。

`<video>` 标签拖动进度条依赖 HTTP Range，所以这里必须自己实现 206 响应，
不能直接用 FileResponse。
"""

from __future__ import annotations

import asyncio
import logging
import mimetypes
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

from ..core.config import FRAME_DIR, UPLOAD_DIR
from ..dependencies import StoreDep
from ..services import ffmpeg as ff

log = logging.getLogger("api.media")
router = APIRouter(prefix="/api/media", tags=["media"])

RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")
STREAM_CHUNK = 1 << 20


@router.get("/upload/{filename}", summary="视频流（支持 Range）")
async def media_upload(filename: str, request: Request):
    path = UPLOAD_DIR / Path(filename).name
    if not path.is_file():
        raise HTTPException(404, "文件不存在")
    return _range_response(path, request)


@router.get("/frame/{job_id}/{chunk}/{filename}", summary="抽帧图片")
async def media_frame(job_id: str, chunk: str, filename: str) -> FileResponse:
    path = FRAME_DIR / Path(job_id).name / Path(chunk).name / Path(filename).name
    if not path.is_file():
        raise HTTPException(404, "帧不存在")
    return FileResponse(
        path,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@router.get("/thumb/{job_id}", summary="任务缩略图")
async def media_thumb(job_id: str, jobs: StoreDep) -> FileResponse:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "任务不存在")

    # 有抽帧结果就直接用第一帧
    if job.result and job.result.frame_urls:
        parts = job.result.frame_urls[0].split("/api/media/frame/", 1)[-1].split("/")
        path = FRAME_DIR.joinpath(*[Path(p).name for p in parts])
        if path.is_file():
            return FileResponse(path, media_type="image/jpeg")

    # 还没抽帧 → 现抽一张
    name = Path(job.video_url).name if job.video_url else ""
    src = UPLOAD_DIR / name
    if not src.is_file():
        raise HTTPException(404, "视频不存在")
    out = FRAME_DIR / f"thumb_{job_id}.jpg"
    got = await asyncio.to_thread(ff.make_thumbnail, src, out, 1.0)
    if not got:
        raise HTTPException(500, "缩略图生成失败")
    return FileResponse(got, media_type="image/jpeg")


@router.get("/frames/{job_id}", summary="任务全部抽帧（给前端做帧带）")
async def media_frames(job_id: str, jobs: StoreDep) -> dict:
    job = jobs.get(job_id)
    if not job or not job.result:
        raise HTTPException(404, "任务结果不存在")
    return {
        "count": job.result.frames_used,
        "urls": job.result.frame_urls,
        "shots": [s.model_dump() for s in job.result.shots],
    }


def _range_response(path: Path, request: Request):
    """实现 HTTP Range，让 <video> 能拖动进度条。"""
    media_type = mimetypes.guess_type(path.name)[0] or "video/mp4"
    file_size = path.stat().st_size
    range_header = request.headers.get("range")

    if not range_header:
        return FileResponse(
            path,
            media_type=media_type,
            headers={"Accept-Ranges": "bytes", "Content-Length": str(file_size)},
        )

    m = RANGE_RE.match(range_header)
    if not m:
        return FileResponse(path, media_type=media_type, headers={"Accept-Ranges": "bytes"})

    start = int(m.group(1)) if m.group(1) else 0
    end = int(m.group(2)) if m.group(2) else file_size - 1
    start = max(0, min(start, file_size - 1))
    end = max(start, min(end, file_size - 1))
    length = end - start + 1

    def _iter():
        remaining = length
        with path.open("rb") as fh:
            fh.seek(start)
            while remaining > 0:
                chunk = fh.read(min(STREAM_CHUNK, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    return StreamingResponse(
        _iter(),
        status_code=206,
        media_type=media_type,
        headers={
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(length),
        },
    )
