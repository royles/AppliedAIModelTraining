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
    os.environ.pop("RAY_AUTH_TOKEN", None)


def _read_token_file(path: Path) -> str | None:
    try:
        if path.is_file():
            token = path.read_text(encoding="utf-8").strip()
            return token or None
    except OSError:
        return None
    return None


def _read_persisted_bundle() -> tuple[str | None, str | None]:
    """Return (gcs_address, token) from the project auth file."""
    path = project_gcs_auth_token_path()
    if not path.is_file():
        return None, None
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None, None
    gcs: str | None = None
    token: str | None = None
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("RAY_GCS_ADDRESS="):
            gcs = line.split("=", 1)[1].strip() or None
        elif line.startswith("RAY_AUTH_TOKEN="):
            token = line.split("=", 1)[1].strip() or None
    if token is None:
        legacy = raw.strip()
        if legacy and "\n" not in legacy and "=" not in legacy:
            token = legacy
    return gcs, token


def _parse_cli_token(stdout: str) -> str | None:
    for line in reversed((stdout or "").splitlines()):
        token = line.strip()
        if token and not token.startswith("Usage"):
            return token
    return None


def _read_head_session_auth_token() -> str | None:
    """Token Ray writes on the session after ``ray start --head``."""
    return _read_token_file(Path.home() / ".ray" / "auth_token")


def _persist_gcs_auth_token(token: str, gcs_address: str) -> None:
    token = token.strip()
    gcs_address = (gcs_address or "").strip()
    if not token:
        return
    os.environ["RAY_AUTH_TOKEN"] = token
    project_path = project_gcs_auth_token_path()
    project_path.parent.mkdir(parents=True, exist_ok=True)
    body = f"RAY_GCS_ADDRESS={gcs_address}\nRAY_AUTH_TOKEN={token}\n"
    project_path.write_text(body, encoding="utf-8")
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


def _invalidate_if_wrong_head(gcs_address: str) -> None:
    persisted_gcs, _ = _read_persisted_bundle()
    if persisted_gcs and persisted_gcs != gcs_address:
        logger.warning(
            "Dropping cached GCS auth (file was for %s, live head is %s)",
            persisted_gcs,
            gcs_address,
        )
        invalidate_stored_gcs_auth_tokens()


def load_gcs_auth_token_from_disk() -> str | None:
    """Load token from env or known files (does not contact Ray). Prefer ``bind_gcs_auth``."""
    token = os.environ.get("RAY_AUTH_TOKEN", "").strip()
    if token:
        return token
    _, token = _read_persisted_bundle()
    if token:
        os.environ["RAY_AUTH_TOKEN"] = token
        return token
    for path in (
        os.environ.get("RAY_AUTH_TOKEN_PATH", "").strip(),
        os.environ.get("PLATFORM_RAY_GCS_AUTH_FILE", "").strip(),
        str(Path.home() / ".ray" / "auth_token"),
    ):
        if not path:
            continue
        found = _read_token_file(Path(path))
        if found:
            os.environ["RAY_AUTH_TOKEN"] = found
            return found
    return None


def ray_gcs_status_ok(gcs_address: str, token: str | None, timeout_sec: float = 25.0) -> bool:
    """True when ``ray status`` succeeds against GCS with the given token (if any)."""
    env = os.environ.copy()
    env["RAY_ADDRESS"] = gcs_address
    if token:
        env["RAY_AUTH_TOKEN"] = token
    else:
        env.pop("RAY_AUTH_TOKEN", None)
    try:
        proc = subprocess.run(
            _ray_cli("status"),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    if proc.returncode != 0:
        return False
    combined = f"{proc.stdout or ''}\n{proc.stderr or ''}".lower()
    if "invalidauthtoken" in combined or "authentication token" in combined:
        return False
    return True


def _fetch_token_from_head_cli(gcs_address: str, *, generate_if_missing: bool) -> tuple[str | None, str]:
    env = os.environ.copy()
    env["RAY_ADDRESS"] = gcs_address
    env.pop("RAY_AUTH_TOKEN", None)

    attempts: list[list[str]] = [["get-auth-token"]]
    if generate_if_missing:
        attempts.append(["get-auth-token", "--generate"])

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
                return token, last_err
        last_err = (proc.stderr or proc.stdout or "")[-400:]
    return None, last_err


def _load_verified_disk_token(gcs_address: str) -> str | None:
    saved_env = os.environ.get("RAY_AUTH_TOKEN")
    try:
        persisted_gcs, persisted_token = _read_persisted_bundle()
        if persisted_token and persisted_gcs == gcs_address:
            if ray_gcs_status_ok(gcs_address, persisted_token):
                os.environ["RAY_AUTH_TOKEN"] = persisted_token
                return persisted_token
        for path in (
            os.environ.get("RAY_AUTH_TOKEN_PATH", "").strip(),
            os.environ.get("PLATFORM_RAY_GCS_AUTH_FILE", "").strip(),
            str(project_gcs_auth_token_path()),
            str(Path.home() / ".ray" / "auth_token"),
        ):
            if not path:
                continue
            candidate = _read_token_file(Path(path))
            if candidate and ray_gcs_status_ok(gcs_address, candidate):
                os.environ["RAY_AUTH_TOKEN"] = candidate
                return candidate
    finally:
        if saved_env is None:
            os.environ.pop("RAY_AUTH_TOKEN", None)
        else:
            os.environ["RAY_AUTH_TOKEN"] = saved_env
    return None


def sync_ray_gcs_auth_token(
    gcs_address: str,
    *,
    generate_if_missing: bool = True,
    required: bool = False,
) -> str | None:
    gcs_address = (gcs_address or "").strip()
    if not gcs_address:
        if required:
            raise RuntimeError("Cannot sync Ray GCS auth token: GCS address is empty")
        return load_gcs_auth_token_from_disk()

    _invalidate_if_wrong_head(gcs_address)

    env_token = os.environ.get("RAY_AUTH_TOKEN", "").strip()
    if env_token and ray_gcs_status_ok(gcs_address, env_token):
        _persist_gcs_auth_token(env_token, gcs_address)
        return env_token

    token = _read_head_session_auth_token()
    if token and ray_gcs_status_ok(gcs_address, token):
        _persist_gcs_auth_token(token, gcs_address)
        return token

    token, last_err = _fetch_token_from_head_cli(
        gcs_address, generate_if_missing=generate_if_missing
    )

    if not token:
        token = _load_verified_disk_token(gcs_address)

    if not token:
        invalidate_stored_gcs_auth_tokens()
        token, retry_err = _fetch_token_from_head_cli(
            gcs_address, generate_if_missing=True
        )
        if retry_err:
            last_err = retry_err or last_err

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

    _persist_gcs_auth_token(token, gcs_address)
    return token


def bind_gcs_auth_for_address(gcs_address: str, *, required: bool = True) -> str:
    """Sync token for ``gcs_address`` and set process env — use before workers or ``ray.init``."""
    token = sync_ray_gcs_auth_token(gcs_address, required=required)
    if not token:
        raise RuntimeError(f"GCS auth token missing for {gcs_address}")
    return token


def require_ray_gcs_auth_token(gcs_address: str) -> str:
    return bind_gcs_auth_for_address(gcs_address, required=True)
