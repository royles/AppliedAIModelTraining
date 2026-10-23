"""Lightweight progress heartbeats for long CPU/IO phases (framework log + console)."""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Callable, Iterator

from app.services.log_flair import heartbeat_line

if TYPE_CHECKING:
    from app.services.download_progress import DownloadProgressState

logger = logging.getLogger(__name__)

LogFn = Callable[[str], None]


def _heartbeat_interval_sec() -> float:
    raw = os.environ.get("PLATFORM_PROVISION_HEARTBEAT_SEC", "30").strip()
    try:
        return max(10.0, float(raw))
    except ValueError:
        return 30.0


@contextmanager
def provisioning_heartbeat(
    log: LogFn | None,
    label: str,
    *,
    interval_sec: float | None = None,
    download_progress: DownloadProgressState | None = None,
) -> Iterator[None]:
    """
    Emit ``[..] {label} — still running (Ns)`` every ``interval_sec`` until the block exits.

    Optional ``download_progress`` appends live row counts (HF/JSONL) — not Ray head checks.

    Safe when ``log`` is None (console-only jobs / tests).
    """
    every = interval_sec if interval_sec is not None else _heartbeat_interval_sec()
    if not log or every <= 0:
        yield
        return

    stop = threading.Event()
    started = time.monotonic()

    def _beat() -> None:
        while not stop.wait(every):
            elapsed = int(time.monotonic() - started)
            detail = ""
            if download_progress is not None:
                snap = download_progress.snapshot()
                if snap:
                    detail = snap
            line = heartbeat_line(label, elapsed, detail)
            try:
                log(line)
            except Exception:  # noqa: BLE001
                logger.debug("Heartbeat log failed", exc_info=True)
            logger.info("%s — still running (%ss)", label, elapsed)

    thread = threading.Thread(
        target=_beat,
        name=f"heartbeat-{label[:24]}",
        daemon=True,
    )
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=2.0)
