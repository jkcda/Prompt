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
    """建表 + 补齐缺失列（幂等）。启动时调用。"""
    from ..models import job as _job_models  # noqa: F401  确保模型已注册到 metadata

    engine = get_engine()
    SQLModel.metadata.create_all(engine)
    _ensure_columns(engine)
    log.info("数据表就绪：%s", ", ".join(SQLModel.metadata.tables.keys()))


def _ensure_columns(engine) -> None:
    """给已存在的表补上模型里新增的列。

    为什么需要：`create_all` 只建新表，不会改老表。而这个项目的结果字段还会继续长
    （例如新增主体登记表），如果不补列，老库一升级就会报 "no such column"。
    这里只做加法——新增列一律可空，不删列、不改类型，因此不会丢数据。
    """
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())

    with engine.begin() as conn:
        for table_name, table in SQLModel.metadata.tables.items():
            if table_name not in existing_tables:
                continue
            have = {c["name"] for c in inspector.get_columns(table_name)}
            for column in table.columns:
                if column.name in have:
                    continue
                if not column.nullable and column.default is None and not column.autoincrement:
                    log.warning("跳过非空无默认值的新列 %s.%s，请手工处理",
                                table_name, column.name)
                    continue
                ddl = f'ALTER TABLE "{table_name}" ADD COLUMN "{column.name}" {column.type.compile(engine.dialect)}'
                conn.execute(text(ddl))
                log.info("补列：%s.%s", table_name, column.name)


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
