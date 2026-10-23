"""Lightweight progress heartbeats for long CPU/IO phases (framework log + console)."""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from typing import Callable, Iterator

from app.services.log_flair import heartbeat_line

logger = logging.getLogger(__name__)

LogFn = Callable[[str], None]

DEFAULT_HEARTBEAT_INTERVAL_SEC = 30.0


@contextmanager
def provisioning_heartbeat(
    log: LogFn | None,
    label: str,
    *,
    interval_sec: float = DEFAULT_HEARTBEAT_INTERVAL_SEC,
) -> Iterator[None]:
    """
    Emit ``[..] {label} — still running (Ns)`` every ``interval_sec`` until the block exits.

    Safe when ``log`` is None (console-only jobs / tests).
    """
    if not log or interval_sec <= 0:
        yield
        return

    stop = threading.Event()
    started = time.monotonic()

    def _beat() -> None:
        while not stop.wait(interval_sec):
            elapsed = int(time.monotonic() - started)
            line = heartbeat_line(label, elapsed)
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
