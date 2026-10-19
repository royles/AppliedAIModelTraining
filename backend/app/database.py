from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings


class Base(DeclarativeBase):
    pass


def _make_engine():
    settings = get_settings()
    connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
    return create_engine(settings.database_url, connect_args=connect_args)


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


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
