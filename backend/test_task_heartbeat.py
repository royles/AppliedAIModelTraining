"""Provisioning heartbeats emit periodic framework lines."""

from __future__ import annotations

import time

from app.services.download_progress import DownloadProgressState
from app.services.task_heartbeat import provisioning_heartbeat


def test_provisioning_heartbeat_emits_during_work() -> None:
    lines: list[str] = []

    def capture(line: str) -> None:
        lines.append(line)

    with provisioning_heartbeat(capture, "Test phase", interval_sec=0.05):
        time.sleep(0.15)

    assert any("still running" in ln for ln in lines)


def test_provisioning_heartbeat_includes_download_progress() -> None:
    lines: list[str] = []
    progress = DownloadProgressState()
    progress.update_rows(12000, 50000)

    def capture(line: str) -> None:
        lines.append(line)

    with provisioning_heartbeat(
        capture, "Dataset download", interval_sec=0.05, download_progress=progress
    ):
        time.sleep(0.15)

    assert any("download 12,000/50,000" in ln for ln in lines)
