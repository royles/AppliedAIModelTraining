"""Provisioning heartbeats emit periodic framework lines."""

from __future__ import annotations

import time

from app.services.task_heartbeat import provisioning_heartbeat


def test_provisioning_heartbeat_emits_during_work() -> None:
    lines: list[str] = []

    def capture(line: str) -> None:
        lines.append(line)

    with provisioning_heartbeat(capture, "Test phase", interval_sec=0.05):
        time.sleep(0.15)

    assert any("still running" in ln for ln in lines)
