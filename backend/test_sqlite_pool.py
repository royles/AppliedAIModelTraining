"""SQLite engine settings for CAI (WAL, modest pool)."""

from __future__ import annotations

from app.database import engine


def test_sqlite_engine_uses_modest_pool_when_sqlite():
    if not str(engine.url).startswith("sqlite"):
        return
    pool = engine.pool
    assert pool.size() <= 5
    assert getattr(pool, "_max_overflow", 0) <= 5
