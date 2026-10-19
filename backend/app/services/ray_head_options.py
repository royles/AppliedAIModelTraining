"""Ray head / State API options for Ray Train v2 (PlacementGroupCleaner)."""

from __future__ import annotations

import os
from urllib.error import URLError
from urllib.request import urlopen


def ray_dashboard_importable() -> bool:
    try:
        import ray.dashboard.dashboard  # noqa: F401
    except ImportError:
        return False
    return True


def ray_start_include_dashboard() -> bool:
    """
    Whether `ray start --head` should pass --include-dashboard=true.

    Ray Train v2's PlacementGroupCleaner polls actor state via ray.util.state (HTTP API).
    With dashboard disabled, those calls fail and spam warnings.
    """
    explicit = os.environ.get("RAY_INCLUDE_DASHBOARD", "").strip().lower()
    if explicit in ("0", "false", "no"):
        return False
    if explicit in ("1", "true", "yes"):
        return True
    return ray_dashboard_importable()


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
