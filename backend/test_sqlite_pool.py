"""SQLite engine settings for CAI (WAL, NullPool)."""

from __future__ import annotations

from sqlalchemy.pool import NullPool

from app.database import engine


def test_sqlite_engine_uses_nullpool_when_sqlite():
    if not str(engine.url).startswith("sqlite"):
        return
    assert isinstance(engine.pool, NullPool)
