"""
Ray 2.59+ GCS authentication token — local only (no external IdP).

Ray stores a cluster auth token when the head starts. Clients (API, CDSW workers)
must send ``RAY_AUTH_TOKEN`` to join GCS. Tokens are refreshed from the live head
via ``ray get-auth-token`` and copied to a project file workers can read.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

from app.config import REPO_ROOT

logger = logging.getLogger(__name__)


def _ray_cli(*args: str) -> list[str]:
    return [sys.executable, "-m", "ray.scripts.scripts", *args]


def project_gcs_auth_token_path() -> Path:
    """Shared via project mount so CDSW workers can read the session head token."""
    return (REPO_ROOT / "data" / ".ray_gcs_auth_token").resolve()


def invalidate_stored_gcs_auth_tokens() -> None:
    """Drop cached tokens (required after ``ray stop`` — old tokens reject on new head)."""
    for path in (project_gcs_auth_token_path(), Path.home() / ".ray" / "auth_token"):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.debug("Could not remove %s", path, exc_info=True)


def _read_token_file(path: Path) -> str | None:
    try:
        if path.is_file():
            token = path.read_text(encoding="utf-8").strip()
            return token or None
    except OSError:
        return None
    return None


def _parse_cli_token(stdout: str) -> str | None:
    for line in reversed((stdout or "").splitlines()):
        token = line.strip()
        if token and not token.startswith("Usage"):
            return token
    return None


def _persist_gcs_auth_token(token: str) -> None:
    token = token.strip()
    if not token:
        return
    os.environ["RAY_AUTH_TOKEN"] = token
    project_path = project_gcs_auth_token_path()
    project_path.parent.mkdir(parents=True, exist_ok=True)
    project_path.write_text(token + "\n", encoding="utf-8")
    try:
        project_path.chmod(0o600)
    except OSError:
        pass
    os.environ["PLATFORM_RAY_GCS_AUTH_FILE"] = str(project_path)
    home_ray = Path.home() / ".ray"
    try:
        home_ray.mkdir(parents=True, exist_ok=True)
        (home_ray / "auth_token").write_text(token + "\n", encoding="utf-8")
    except OSError:
        logger.debug("Could not write ~/.ray/auth_token", exc_info=True)


def load_gcs_auth_token_from_disk() -> str | None:
    """Load token from env or known files (does not contact Ray)."""
    token = os.environ.get("RAY_AUTH_TOKEN", "").strip()
    if token:
        return token
    for path in (
        os.environ.get("RAY_AUTH_TOKEN_PATH", "").strip(),
        os.environ.get("PLATFORM_RAY_GCS_AUTH_FILE", "").strip(),
        str(project_gcs_auth_token_path()),
        str(Path.home() / ".ray" / "auth_token"),
    ):
        if not path:
            continue
        found = _read_token_file(Path(path))
        if found:
            os.environ["RAY_AUTH_TOKEN"] = found
            return found
    return None


def sync_ray_gcs_auth_token(
    gcs_address: str,
    *,
    generate_if_missing: bool = True,
    required: bool = False,
) -> str | None:
    """
    Refresh ``RAY_AUTH_TOKEN`` from the running Ray head at ``gcs_address``.

    No external services — only the local ``ray`` CLI talking to GCS on the session.
    """
    gcs_address = (gcs_address or "").strip()
    if not gcs_address:
        if required:
            raise RuntimeError("Cannot sync Ray GCS auth token: GCS address is empty")
        return load_gcs_auth_token_from_disk()

    env = os.environ.copy()
    env["RAY_ADDRESS"] = gcs_address

    attempts: list[list[str]] = [["get-auth-token"]]
    if generate_if_missing:
        attempts.append(["get-auth-token", "--generate"])

    token: str | None = None
    last_err = ""
    for args in attempts:
        try:
            proc = subprocess.run(
                _ray_cli(*args),
                env=env,
                capture_output=True,
                text=True,
                timeout=45,
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            last_err = str(exc)
            continue
        if proc.returncode == 0:
            token = _parse_cli_token(proc.stdout or "")
            if token:
                break
        last_err = (proc.stderr or proc.stdout or "")[-400:]

    if not token:
        token = load_gcs_auth_token_from_disk()

    if not token:
        msg = (
            f"Ray GCS auth token unavailable for head {gcs_address}. "
            "Run `ray stop --force` and `python3 scripts/start.py` on the master session. "
            "Ray 2.59+ requires this token for workers and `ray status` (no external IdP)."
        )
        if last_err:
            msg += f" Last ray get-auth-token: {last_err}"
        if required:
            raise RuntimeError(msg)
        logger.warning(msg)
        return None

    _persist_gcs_auth_token(token)
    return token


def require_ray_gcs_auth_token(gcs_address: str) -> str:
    token = sync_ray_gcs_auth_token(gcs_address, required=True)
    assert token
    return token
