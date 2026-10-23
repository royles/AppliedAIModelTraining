"""Thread-safe download row counts for provisioning heartbeats (no Ray logging)."""

from __future__ import annotations

import threading


class DownloadProgressState:
    """Updated during HF/JSONL loads; read from heartbeat threads."""

    __slots__ = ("_lock", "_detail")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._detail = ""

    def update_rows(self, count: int, limit: int | None = None, *, unit: str = "rows") -> None:
        with self._lock:
            if limit is not None:
                self._detail = f"download {count:,}/{limit:,} {unit}"
            else:
                self._detail = f"download {count:,} {unit}"

    def snapshot(self) -> str | None:
        with self._lock:
            return self._detail or None
