"""日志配置。

统一格式，并按模块分级：第三方库（httpx / urllib3 / asyncio）压到 WARNING，
否则每次模型调用都会刷一屏噪音。
"""

from __future__ import annotations

import logging
import sys

from .config import get_settings

FORMAT = "%(asctime)s %(levelname)-7s %(name)-16s %(message)s"
DATEFMT = "%H:%M:%S"

NOISY = (
    "httpx", "httpcore", "urllib3", "asyncio",
    "multipart", "python_multipart", "watchfiles", "uvicorn.access",
)


def setup_logging(level: str | None = None) -> None:
    settings = get_settings()
    resolved = (level or ("DEBUG" if settings.debug else "INFO")).upper()

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(resolved)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(FORMAT, datefmt=DATEFMT))
    root.addHandler(handler)

    for name in NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)
