"""全局配置：pydantic-settings 读取 .env，附带路径与 ffmpeg 解析。"""

from __future__ import annotations

import os
import shutil
import sys
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# 目录层级：app/core/config.py -> app/core -> app -> backend -> 项目根
CORE_DIR = Path(__file__).resolve().parent
APP_DIR = CORE_DIR.parent
BACKEND_DIR = APP_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
FRAME_DIR = DATA_DIR / "frames"
TMP_DIR = DATA_DIR / "tmp"
RUNTIME_DIR = BACKEND_DIR / "runtime"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

for _d in (DATA_DIR, UPLOAD_DIR, FRAME_DIR, TMP_DIR, RUNTIME_DIR):
    _d.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- 服务 ----
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: str = "*"
    debug: bool = False

    # ---- 持久化 ----
    # SQLite 单文件。相对路径按 data/ 解析，也可写绝对路径。
    database_url: str = "sqlite:///app.db"
    # 保留多少天的任务历史（0 = 永久保留）
    history_keep_days: int = 30

    # ---- 多模态模型 ----
    vlm_api_key: str = ""
    vlm_base_url: str = "https://api-inference.modelscope.cn/v1"
    vlm_model: str = "Qwen/Qwen2.5-VL-72B-Instruct"
    vlm_timeout: int = 180
    vlm_concurrency: int = 3

    # ---- 语音转写 ----
    asr_api_key: str = ""
    asr_base_url: str = ""
    asr_model: str = "whisper-1"

    # ---- ffmpeg ----
    ffmpeg_path: str = ""
    ffmpeg_dir: str = ""

    # ---- 抽帧预算 ----
    max_total_frames: int = 48
    max_frames_per_shot: int = 3
    long_shot_seconds: float = 5.0
    frame_long_edge: int = 896
    frame_jpeg_quality: int = 82

    # ---- 镜头分割 ----
    scene_threshold: float = 0.30
    min_shot_seconds: float = 0.8

    # ---- 长视频分块 ----
    chunk_threshold_seconds: float = 90.0
    chunk_seconds: float = 60.0
    chunk_overlap_seconds: float = 2.0
    max_duration_seconds: float = 1800.0

    # ---- 上传 ----
    max_upload_mb: int = 500

    # ---- 抓取 ----
    cookies_from_browser: str = ""
    cookies_file: str = ""

    # ---------- 派生属性 ----------
    @property
    def cors_origin_list(self) -> list[str]:
        raw = (self.cors_origins or "").strip()
        if not raw or raw == "*":
            return ["*"]
        return [x.strip() for x in raw.split(",") if x.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def asr_enabled(self) -> bool:
        return bool(self.asr_base_url and self.asr_api_key)

    @property
    def vlm_enabled(self) -> bool:
        return bool(self.vlm_api_key)


# ---------------------------------------------------------------------------
# ffmpeg / ffprobe 自动发现
# ---------------------------------------------------------------------------

def _candidates(names: tuple[str, ...]) -> list[Path]:
    """按优先级列出可能的可执行文件路径。"""
    exe = ".exe" if os.name == "nt" else ""
    out: list[Path] = []

    s = get_settings()

    # 1. 显式配置
    if s.ffmpeg_path and "ffmpeg" in names[0]:
        out.append(Path(s.ffmpeg_path))
    if s.ffmpeg_dir:
        for n in names:
            out.append(Path(s.ffmpeg_dir) / f"{n}{exe}")

    # 2. 项目内 runtime 目录（可选的自带二进制位置）
    for n in names:
        out.append(RUNTIME_DIR / f"{n}{exe}")

    # 3. 系统 PATH
    for n in names:
        found = shutil.which(n)
        if found:
            out.append(Path(found))

    # 4. 常见安装位置 / 复用已有项目里的静态二进制
    common_dirs = [
        PROJECT_ROOT / "bin",
        Path("D:/nexus/aiconnent/server/node_modules/ffmpeg-static"),
        Path("C:/ffmpeg/bin"),
        Path("D:/ffmpeg/bin"),
        Path("C:/Program Files/ffmpeg/bin"),
        Path("D:/Program Files/ffmpeg/bin"),
        Path("/usr/bin"),
        Path("/usr/local/bin"),
        Path("/opt/homebrew/bin"),
    ]
    for d in common_dirs:
        for n in names:
            out.append(d / f"{n}{exe}")

    return out


@lru_cache(maxsize=1)
def resolve_ffmpeg() -> str | None:
    for p in _candidates(("ffmpeg",)):
        try:
            if p.is_file():
                return str(p)
        except OSError:
            continue
    return None


@lru_cache(maxsize=1)
def resolve_ffprobe() -> str | None:
    """ffprobe 通常缺失；找不到时上层用 `ffmpeg -i` 解析替代。"""
    for p in _candidates(("ffprobe",)):
        try:
            if p.is_file():
                return str(p)
        except OSError:
            continue
    return None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def refresh_settings() -> Settings:
    """清缓存后重新读取 .env（供 /api/settings 热更新使用）。"""
    get_settings.cache_clear()
    resolve_ffmpeg.cache_clear()
    resolve_ffprobe.cache_clear()
    return get_settings()


def ffmpeg_required() -> str:
    path = resolve_ffmpeg()
    if not path:
        raise RuntimeError(
            "未找到 ffmpeg。请任选其一：\n"
            "  1) 把 ffmpeg 加入系统 PATH；\n"
            "  2) 在 backend/.env 里设置 FFMPEG_PATH=D:/path/to/ffmpeg.exe；\n"
            "  3) 把 ffmpeg.exe 放到 backend/runtime/ 目录下。"
        )
    return path


def python_exe() -> str:
    return sys.executable
