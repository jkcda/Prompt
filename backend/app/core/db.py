"""SQLite 持久化层：引擎、会话、建表。

为什么用 SQLite + SQLModel：
    - 单文件数据库，不需要额外起服务，部署就是拷一个 .db；
    - SQLModel 把 Pydantic 模型和表模型合一，接口层能直接复用；
    - 反推结果体量大且结构会变（观察字段可能增删），所以结果类字段用 JSON 列，
      只把「需要筛选/排序」的字段建成真实列（state / created_at / format / title）。

表结构见 `app/models/job.py`。写入逻辑在 `app/services/storage.py`。
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

from sqlmodel import Session, SQLModel, create_engine

from .config import DATA_DIR, get_settings

log = logging.getLogger("db")

_engine = None


def db_path() -> Path:
    s = get_settings()
    p = Path(s.database_url.replace("sqlite:///", "")) if s.database_url.startswith("sqlite:///") \
        else DATA_DIR / "app.db"
    if not p.is_absolute():
        p = DATA_DIR / p
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def get_engine():
    global _engine
    if _engine is None:
        path = db_path()
        _engine = create_engine(
            f"sqlite:///{path.as_posix()}",
            echo=False,
            connect_args={"check_same_thread": False},
        )
        log.info("SQLite: %s", path)
    return _engine


def init_db() -> None:
    """建表（幂等）。启动时调用。"""
    from ..models import job as _job_models  # noqa: F401  确保模型已注册到 metadata

    SQLModel.metadata.create_all(get_engine())
    log.info("数据表就绪：%s", ", ".join(SQLModel.metadata.tables.keys()))


def get_session() -> Iterator[Session]:
    """FastAPI 依赖：每个请求一个会话。"""
    with Session(get_engine()) as session:
        yield session


def reset_engine() -> None:
    """配置变更后重建引擎（切换数据库路径时用）。"""
    global _engine
    if _engine is not None:
        _engine.dispose()
    _engine = None
