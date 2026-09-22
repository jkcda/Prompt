"""FastAPI 共享依赖。

放在这里的都是「多个路由模块都要用」的东西，路由自己不该重复构造。
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends
from sqlmodel import Session

from .core.config import Settings, get_settings
from .core.db import get_session
from .services.jobs import JobStore, store
from .services.vlm import VLMClient


def settings_dep() -> Settings:
    return get_settings()


def store_dep() -> JobStore:
    return store


def vlm_dep() -> VLMClient:
    """每次调用新建客户端（内部是轻量的信号量 + httpx），避免跨请求共享状态。"""
    return VLMClient()


SettingsDep = Annotated[Settings, Depends(settings_dep)]
StoreDep = Annotated[JobStore, Depends(store_dep)]
VLMDDep = Annotated[VLMClient, Depends(vlm_dep)]


def session_dep() -> Iterator[Session]:
    yield from get_session()


SessionDep = Annotated[Session, Depends(session_dep)]
