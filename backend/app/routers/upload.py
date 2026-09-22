"""上传接口。"""

from __future__ import annotations

import asyncio
import logging
import re
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
    """流式落盘，边写边校验大小，避免把整个文件读进内存。

    落盘文件名刻意保持纯 ASCII：中文文件名虽然更好认，但会出现在 URL 里，
    一旦某一环没做百分号编码（curl、部分代理、Content-Disposition）就会 400。
    原始中文名放在响应体的 `name` 字段里给前端展示，两边各取所需。
    """
    raw_name = file.filename or "video.mp4"
    ext = Path(raw_name).suffix.lower()
    if ext not in VIDEO_EXTS:
        raise HTTPException(
            400,
            f"不支持的视频格式：{ext or '未知'}。支持 {', '.join(sorted(VIDEO_EXTS))}",
        )

    file_id = _make_file_id(Path(raw_name).stem, ext)
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


def _make_file_id(stem: str, ext: str) -> str:
    """生成纯 ASCII 的落盘文件名，尽量保留可读的英文/数字片段。"""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_")[:32]
    prefix = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
    return f"{prefix}_{slug}{ext}" if slug else f"{prefix}{ext}"
