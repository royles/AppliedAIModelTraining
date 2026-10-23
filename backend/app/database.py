from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings


class Base(DeclarativeBase):
    pass


def _configure_sqlite_connection(dbapi_conn) -> None:
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA busy_timeout=60000")
    cursor.close()


def _make_engine():
    settings = get_settings()
    url = settings.database_url
    connect_args = {"check_same_thread": False, "timeout": 60} if url.startswith("sqlite") else {}
    kwargs: dict = {"connect_args": connect_args, "pool_pre_ping": True}
    if url.startswith("sqlite"):
        # SQLite is single-writer; a large QueuePool + long-lived job sessions starves /api polls.
        kwargs["pool_size"] = 5
        kwargs["max_overflow"] = 5
        kwargs["pool_timeout"] = 30
        kwargs["pool_recycle"] = 300
    engine = create_engine(settings.database_url, **kwargs)
    if url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _on_sqlite_connect(dbapi_conn, _connection_record) -> None:
            _configure_sqlite_connection(dbapi_conn)

    return engine


engine = _make_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def _sqlite_add_column_if_missing(table: str, column: str, ddl: str) -> None:
    if not str(engine.url).startswith("sqlite"):
        return
    with engine.connect() as conn:
        rows = conn.exec_driver_sql(f"PRAGMA table_info({table})").fetchall()
        names = {row[1] for row in rows}
        if column not in names:
            conn.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN {ddl}")
            conn.commit()


def init_db() -> None:
    from app import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    _sqlite_add_column_if_missing("training_jobs", "config_json", "config_json TEXT")
    _sqlite_add_column_if_missing("training_jobs", "log_json", "log_json TEXT")
    _sqlite_add_column_if_missing(
        "training_jobs", "task_version_id", "task_version_id INTEGER REFERENCES task_versions(id)"
    )
    _sqlite_add_column_if_missing(
        "platform_settings",
        "non_distributed_low_memory_mode",
        "non_distributed_low_memory_mode INTEGER DEFAULT 0",
    )
    for col, ddl in (
        ("lifecycle_state", "lifecycle_state VARCHAR(32) DEFAULT 'trained'"),
        ("display_name", "display_name VARCHAR(255)"),
        ("description", "description TEXT"),
        ("metadata_json", "metadata_json TEXT"),
        ("model_card_markdown", "model_card_markdown TEXT"),
        ("benchmark_json", "benchmark_json TEXT"),
        ("parent_model_id", "parent_model_id INTEGER"),
        ("updated_at", "updated_at DATETIME"),
    ):
        _sqlite_add_column_if_missing("trained_models", col, ddl)
    with engine.connect() as conn:
        conn.exec_driver_sql(
            "UPDATE trained_models SET lifecycle_state = 'trained' "
            "WHERE lifecycle_state IS NULL OR lifecycle_state = ''"
        )
        conn.commit()
    from app.database import SessionLocal
    from app.services.project_ray_settings import seed_missing_project_ray_settings

    db = SessionLocal()
    try:
        seed_missing_project_ray_settings(db)
    finally:
        db.close()


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """Short-lived session (use during long Ray waits instead of holding one open)."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
