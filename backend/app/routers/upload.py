"""上传接口。"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from ..core.config import UPLOAD_DIR
from ..dependencies import SettingsDep
from ..schemas import UploadResponse
from ..services import ffmpeg as ff

log = logging.getLogger("api.upload")
router = APIRouter(prefix="/api", tags=["upload"])

VIDEO_EXTS = {
    ".mp4", ".mov", ".mkv", ".webm", ".avi", ".flv",
    ".m4v", ".ts", ".wmv", ".mpg", ".mpeg", ".3gp",
}
CHUNK = 1 << 20  # 1MB


@router.post("/upload", response_model=UploadResponse, summary="上传视频")
async def upload_video(settings: SettingsDep, file: UploadFile = File(...)) -> UploadResponse:
    """流式落盘，边写边校验大小，避免把整个文件读进内存。"""
    raw_name = file.filename or "video.mp4"
    ext = Path(raw_name).suffix.lower()
    if ext not in VIDEO_EXTS:
        raise HTTPException(
            400,
            f"不支持的视频格式：{ext or '未知'}。支持 {', '.join(sorted(VIDEO_EXTS))}",
        )

    stem = _safe_stem(Path(raw_name).stem)
    file_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}_{stem}{ext}"
    dst = UPLOAD_DIR / file_id

    total = 0
    limit = settings.max_upload_bytes
    try:
        with dst.open("wb") as fh:
            while True:
                chunk = await file.read(CHUNK)
                if not chunk:
                    break
                total += len(chunk)
                if total > limit:
                    fh.close()
                    dst.unlink(missing_ok=True)
                    raise HTTPException(413, f"文件超过 {settings.max_upload_mb}MB 上限")
                fh.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        dst.unlink(missing_ok=True)
        log.exception("保存上传文件失败")
        raise HTTPException(500, f"保存上传文件失败：{exc}") from exc

    if total == 0:
        dst.unlink(missing_ok=True)
        raise HTTPException(400, "上传的文件为空")

    info = await asyncio.to_thread(ff.probe, dst)
    log.info("上传完成：%s（%.1fMB，%.1fs）", raw_name, total / 1048576, info.duration)

    return UploadResponse(
        file_id=file_id,
        name=raw_name,
        size=total,
        video_url=f"/api/media/upload/{file_id}",
        duration=info.duration,
        width=info.width,
        height=info.height,
        fps=info.fps,
        has_audio=info.has_audio,
    )


def _safe_stem(stem: str) -> str:
    """保留中文，只替换文件系统不安全的字符，并限长。"""
    bad = '<>:"/\\|?*\x00-\x1f'
    cleaned = "".join("_" if ch in bad else ch for ch in stem).strip(" .")
    return cleaned[:60] or "video"
