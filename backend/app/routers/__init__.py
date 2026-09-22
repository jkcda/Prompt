"""路由聚合。

每个子模块导出一个 `router`，这里统一汇总后由 main.py 一次性挂载，
避免在 main.py 里堆一长串 include_router。
"""

from fastapi import APIRouter

from . import analyze, jobs, media, system, upload

api_router = APIRouter()
api_router.include_router(system.router)
api_router.include_router(upload.router)
api_router.include_router(analyze.router)
api_router.include_router(jobs.router)
api_router.include_router(media.router)

__all__ = ["api_router"]
