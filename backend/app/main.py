"""FastAPI 应用入口。

开发：
    cd backend
    fastapi dev                    # 读 pyproject.toml 的 entrypoint，带热重载
    # 或
    uvicorn app.main:app --reload

生产：
    fastapi run                    # 或 uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .core.auth import BasicAuthMiddleware
from .core.config import FRONTEND_DIR, get_settings, resolve_ffmpeg, resolve_ffprobe
from .core.db import init_db
from .core.logging import setup_logging
from .routers import api_router
from .services.jobs import store
from .services.storage import cleanup_old, mark_interrupted

setup_logging()
log = logging.getLogger("main")

# 前端构建产物（Vue + Vite 的 dist）。开发时前端跑在 Vite 上，这里不用管。
FRONTEND_DIST = FRONTEND_DIR / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()

    log.info("=" * 64)
    log.info("视频提示词反推服务")
    log.info("  ffmpeg   : %s", resolve_ffmpeg() or "!! 未找到，请设置 FFMPEG_PATH")
    log.info("  ffprobe  : %s", resolve_ffprobe() or "(缺失，用 ffmpeg -i 解析替代)")
    log.info("  视觉模型 : %s @ %s", s.vlm_model, s.vlm_base_url)
    log.info("  模型密钥 : %s", "已配置" if s.vlm_api_key else "!! 未配置 VLM_API_KEY")
    log.info("  语音转写 : %s", "已配置" if s.asr_enabled else "未配置（跳过音频维度）")
    log.info("  抽帧预算 : %d 帧 / 每镜最多 %d 帧", s.max_total_frames, s.max_frames_per_shot)
    log.info("  镜头阈值 : scene=%.2f / 最短 %.1fs", s.scene_threshold, s.min_shot_seconds)
    log.info("  前端产物 : %s", FRONTEND_DIST if FRONTEND_DIST.is_dir() else "(未构建)")
    log.info("  接口文档 : http://127.0.0.1:%d/docs", s.port)
    log.info("=" * 64)

    if not resolve_ffmpeg():
        log.warning("未找到 ffmpeg —— 抽帧不可用。把 ffmpeg.exe 放进 backend/runtime/ "
                    "或在 backend/.env 设置 FFMPEG_PATH。")

    init_db()
    # 上一次进程如果是被 kill 的，库里会留下永远 running 的僵尸任务
    stale = mark_interrupted()
    if stale:
        log.info("已收掉 %d 个被中断的任务", stale)
    removed = cleanup_old(s.history_keep_days)
    if removed:
        log.info("已清理 %d 条过期历史", removed)

    yield

    log.info("服务关闭。剩余任务：%s", store.stats())


app = FastAPI(
    title="视频提示词反推",
    description=(
        "上传视频或从 B站/抖音抓取，用 ffmpeg 做镜头分割 + 自适应抽帧，"
        "经两阶段多模态反推生成可直接使用的视频生成提示词。"
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 访问认证。**没设 AUTH_PASSWORD 就不挂中间件** —— 本地开发不该被登录框挡路。
# 对外暴露前必须设上，否则谁能连上就能白嫖你的 VLM 额度。
if _settings.auth_password:
    app.add_middleware(
        BasicAuthMiddleware,
        username=_settings.auth_username,
        password=_settings.auth_password,
    )
    log.info("已启用访问认证（用户名 %s）", _settings.auth_username)
else:
    log.warning(
        "未启用访问认证（AUTH_PASSWORD 为空）—— 本地开发没问题，"
        "对外暴露前务必设置，否则任何人都能提交任务、烧你的 VLM 额度"
    )

app.include_router(api_router)


@app.get("/", include_in_schema=False)
async def index():
    """有前端构建产物就托管它，否则引导到接口文档。"""
    page = FRONTEND_DIST / "index.html"
    if page.is_file():
        return FileResponse(page, media_type="text/html")
    return JSONResponse({
        "name": "视频提示词反推 API",
        "docs": "/docs",
        "hint": "前端未构建。开发时请另行运行 `npm run dev`（frontend/），"
                "或执行 `npm run build` 让后端直接托管。",
    })


@app.get("/{full_path:path}", include_in_schema=False)
async def spa_fallback(full_path: str):
    """前端用 history 路由，直接访问 /job/xxx 这种深链要回落到 index.html。

    只兜底非 /api 路径，且优先返回真实存在的静态文件。
    """
    if full_path.startswith("api/") or full_path in ("docs", "redoc", "openapi.json"):
        raise HTTPException(404, "Not Found")

    candidate = FRONTEND_DIST / full_path
    if candidate.is_file() and FRONTEND_DIST in candidate.resolve().parents:
        return FileResponse(candidate)

    page = FRONTEND_DIST / "index.html"
    if page.is_file():
        return FileResponse(page, media_type="text/html")
    raise HTTPException(404, "前端未构建")


if FRONTEND_DIST.is_dir():
    # 静态资源；前端用 history 模式时 404 由上面的 spa_fallback 兜底
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")


def main() -> None:
    """供 `python -m app.main` 直接启动。"""
    import uvicorn

    s = get_settings()
    uvicorn.run("app.main:app", host=s.host, port=s.port, reload=s.debug, log_level="info")


if __name__ == "__main__":
    main()
