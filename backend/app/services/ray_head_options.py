"""Ray head / State API options for Ray Train v2 (PlacementGroupCleaner)."""

from __future__ import annotations

import os
from urllib.error import URLError
from urllib.request import urlopen


def ray_dashboard_importable() -> bool:
    """
    True only when the dashboard can actually start (not just ``ray.dashboard`` present).

    Minimal ``pip install ray`` often ships dashboard stubs but omits extras such as
    ``aiohttp_cors``, which makes ``ray start --head --include-dashboard=true`` abort
    the entire head process.
    """
    try:
        import aiohttp_cors  # noqa: F401
        import ray.dashboard.dashboard  # noqa: F401
    except ImportError:
        return False
    return True


def ray_start_include_dashboard() -> bool:
    """
    Whether `ray start --head` should pass --include-dashboard=true.

    Ray Train v2's PlacementGroupCleaner polls actor state via ray.util.state (HTTP API).
    With dashboard disabled, those calls fail and spam warnings.

    Never returns True unless ``ray.dashboard`` is importable — otherwise
    ``ray start --head`` fails with "Cannot include dashboard with missing packages."
    """
    explicit = os.environ.get("RAY_INCLUDE_DASHBOARD", "").strip().lower()
    if explicit in ("0", "false", "no"):
        return False
    importable = ray_dashboard_importable()
    if explicit in ("1", "true", "yes"):
        return importable
    return importable


def ray_head_dashboard_cli_args() -> list[str]:
    """CLI flags for ``ray start --head`` (dashboard only when packages exist)."""
    if ray_start_include_dashboard():
        port = int(os.environ.get("RAY_DASHBOARD_PORT", "8265") or "8265")
        return [f"--dashboard-port={port}", "--include-dashboard=true"]
    return ["--include-dashboard=false"]


def state_api_reachable(
    host: str = "127.0.0.1",
    port: int | None = None,
    timeout_sec: float = 2.0,
) -> bool:
    if port is None:
        port = int(os.environ.get("RAY_DASHBOARD_PORT", "8265") or "8265")
    url = f"http://{host}:{port}/api/version"
    try:
        with urlopen(url, timeout=timeout_sec) as resp:
            return 200 <= getattr(resp, "status", 200) < 300
    except (URLError, OSError, ValueError):
        return False
