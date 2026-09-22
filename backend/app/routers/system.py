"""系统类接口：健康检查、运行时配置、格式清单、统计。"""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import APIRouter

from ..core.config import env_file_path, refresh_settings, resolve_ffmpeg, resolve_ffprobe
from ..core.db import db_path
from ..dependencies import SettingsDep, StoreDep
from ..schemas import FormatOption, HealthResponse, PromptModeOption
from ..services import downloader, storage, templates
from ..services.vlm import healthcheck as vlm_healthcheck
from ..services.vlm import list_models

log = logging.getLogger("api.system")
router = APIRouter(prefix="/api", tags=["system"])

# 允许通过接口写入 .env 的键白名单
ENV_WHITELIST = {
    "VLM_API_KEY", "VLM_BASE_URL", "VLM_MODEL", "VLM_CONCURRENCY", "VLM_TIMEOUT",
    "VLM_AUDIO_INPUT", "VLM_MAX_TOKENS",
    "ASR_API_KEY", "ASR_BASE_URL", "ASR_MODEL",
    "MAX_TOTAL_FRAMES", "MAX_FRAMES_PER_SHOT", "LONG_SHOT_SECONDS",
    "FRAME_LONG_EDGE", "SCENE_THRESHOLD", "MIN_SHOT_SECONDS",
    "CHUNK_THRESHOLD_SECONDS", "CHUNK_SECONDS", "MAX_UPLOAD_MB",
    "FFMPEG_PATH", "COOKIES_FROM_BROWSER", "COOKIES_FILE", "YTDLP_FORMAT",
}


@router.get("/health", response_model=HealthResponse, summary="健康检查")
async def health(settings: SettingsDep, jobs: StoreDep) -> HealthResponse:
    return HealthResponse(
        ok=True,
        ffmpeg=resolve_ffmpeg(),
        ffprobe=resolve_ffprobe(),
        vlm_configured=settings.vlm_enabled,
        vlm_model=settings.vlm_model,
        asr_configured=settings.asr_enabled,
        ytdlp=downloader.yt_dlp_available(),
        jobs=jobs.stats(),
        db=str(db_path()),
    )


@router.get("/health/vlm", summary="模型连通性自检（会真的发一张图）")
async def health_vlm() -> dict[str, Any]:
    """真实调用一次模型，确认 key / 模型名 / **图片输入**都可用。

    只发文本的话纯文本模型也能通过，用户会以为配好了，
    真跑反推时才发现收到图片就报错。所以这里会发一张纯红图，
    回答里必须出现 red 才算真能读图。
    """
    return await vlm_healthcheck()


@router.post("/health/vlm", summary="测试一组模型配置（不保存）")
async def test_vlm_config(payload: dict[str, Any]) -> dict[str, Any]:
    """在保存之前先试。用于「切换模型」时逐个试哪个能用。

    传入的 api_key 为空时回退到已保存的 key，这样用户只需要改模型名就能试。
    """
    model = (payload.get("model") or "").strip() or None
    base_url = (payload.get("base_url") or "").strip() or None
    api_key = payload.get("api_key")
    api_key = api_key.strip() if isinstance(api_key, str) and api_key.strip() else None
    return await vlm_healthcheck(model=model, base_url=base_url, api_key=api_key)


@router.get("/models", summary="拉取服务商声明的模型列表")
async def models(base_url: str = "") -> dict[str, Any]:
    """给「切换模型」界面用的候选列表。

    ⚠ 返回的列表不代表账号真实可用范围 —— 服务商常常只返回精选列表，
    里面有些模型没有部署推理服务。必须逐个测（POST /api/health/vlm）才能确定。
    """
    return await list_models(base_url=base_url.strip() or None)


@router.get("/settings", summary="读取当前配置")
async def read_settings(settings: SettingsDep) -> dict[str, Any]:
    return {
        "vlm": {
            "configured": settings.vlm_enabled,
            "base_url": settings.vlm_base_url,
            "model": settings.vlm_model,
            "concurrency": settings.vlm_concurrency,
            "max_tokens": settings.vlm_max_tokens,
            "audio_input": settings.vlm_audio_input,
            "api_key_masked": _mask(settings.vlm_api_key),
        },
        "asr": {
            "configured": settings.asr_enabled,
            "base_url": settings.asr_base_url,
            "model": settings.asr_model,
            "api_key_masked": _mask(settings.asr_api_key),
        },
        "frames": {
            "max_total_frames": settings.max_total_frames,
            "max_frames_per_shot": settings.max_frames_per_shot,
            "long_shot_seconds": settings.long_shot_seconds,
            "long_edge": settings.frame_long_edge,
            "jpeg_quality": settings.frame_jpeg_quality,
        },
        "scene": {
            "threshold": settings.scene_threshold,
            "min_shot_seconds": settings.min_shot_seconds,
        },
        "chunk": {
            "threshold_seconds": settings.chunk_threshold_seconds,
            "seconds": settings.chunk_seconds,
            "overlap_seconds": settings.chunk_overlap_seconds,
        },
        "upload": {"max_mb": settings.max_upload_mb},
        "database": {"url": settings.database_url, "path": str(db_path())},
        "ffmpeg": {"path": resolve_ffmpeg(), "ffprobe": resolve_ffprobe()},
    }


@router.post("/settings", summary="更新配置（写入 backend/.env）")
async def write_settings(payload: dict[str, Any]) -> dict[str, Any]:
    updates = {k.upper(): str(v) for k, v in payload.items() if k.upper() in ENV_WHITELIST}
    if not updates:
        return {"ok": False, "message": "没有可更新的字段", "allowed": sorted(ENV_WHITELIST)}

    env_path = env_file_path()
    lines: list[str] = []
    if env_path.is_file():
        lines = env_path.read_text(encoding="utf-8").splitlines()

    index: dict[str, int] = {}
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        index[stripped.split("=", 1)[0].strip()] = i

    for key, value in updates.items():
        if key in index:
            lines[index[key]] = f"{key}={value}"
        else:
            lines.append(f"{key}={value}")

    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 同时写进当前进程的环境变量。
    # 原因：pydantic-settings 的优先级是「环境变量 > .env 文件」。
    # 如果服务是用 `VLM_MODEL=x uvicorn ...` 这种方式起的，
    # 只写 .env 的话新值会被旧的环境变量盖住 —— 界面上点了保存却毫无反应，
    # 而且没有任何报错，最难查的那种。
    for key, value in updates.items():
        os.environ[key] = value

    refresh_settings()
    log.info("配置已更新：%s", ", ".join(sorted(updates)))
    return {"ok": True, "updated": sorted(updates.keys())}


@router.get("/formats", response_model=list[PromptModeOption], summary="可选反推模式与变体")
async def formats() -> list[PromptModeOption]:
    """按「模式 → 变体」分组返回。

    对外是两种模式：H3 与 Seedance。`generic` 作为工具无关的兜底，
    `primary=False`，前端把它折进「其他格式」而不占主选择位。
    """
    out: list[PromptModeOption] = []
    for mode, variants in templates.MODE_VARIANTS.items():
        options = [
            FormatOption(
                value=value,  # type: ignore[arg-type]
                label=templates.FORMAT_LABELS.get(value, value),
                description=templates.FORMAT_NOTES.get(value, ""),
                mode=mode,  # type: ignore[arg-type]
                default=is_default,
            )
            for value, is_default in variants
        ]
        out.append(PromptModeOption(
            value=mode,  # type: ignore[arg-type]
            label=templates.MODE_LABELS.get(mode, mode),
            description=templates.MODE_NOTES.get(mode, ""),
            primary=templates.MODE_PRIMARY.get(mode, False),
            variants=options,
        ))
    return out


@router.get("/stats", summary="任务统计")
async def stats(jobs: StoreDep) -> dict[str, Any]:
    return {
        "live": jobs.stats(),
        "total_in_db": storage.count_jobs(),
        "db_path": str(db_path()),
    }


def _mask(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}{'*' * 8}{value[-4:]}"
