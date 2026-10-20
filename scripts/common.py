"""Shared helpers for platform lifecycle scripts."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

REPO_ROOT = Path(__file__).resolve().parent.parent
BIND_HOST = "127.0.0.1"
TMUX_CONF = "/exec-daemon/tmux.portal.conf"
RUN_DIR = Path("/tmp/platform-services")
FRONTEND_DIST = REPO_ROOT / "frontend" / "dist"
BACKEND_DIR = REPO_ROOT / "backend"
REQUIREMENTS_FILE = BACKEND_DIR.parent / "backend" / "requirements.txt"
# Modules required before uvicorn / FastAPI can start.
RUNTIME_IMPORTS = ("uvicorn", "fastapi", "sqlalchemy", "pydantic")


def cd_repo() -> None:
    os.chdir(REPO_ROOT)


def ensure_bundled_datasets() -> None:
    """Create data/datasets/starter_corpus.txt when missing (CAI git pull / fresh clones)."""
    datasets_dir = REPO_ROOT / "data" / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    starter = datasets_dir / "starter_corpus.txt"
    if starter.is_file() and starter.stat().st_size >= 10_000:
        return
    generator = REPO_ROOT / "scripts" / "generate_starter_corpus.py"
    if not generator.is_file():
        print(
            f"WARNING: {starter} missing and {generator} not found; "
            "dataset preview may fail until you git pull or add a corpus file.",
            file=sys.stderr,
        )
        return
    print(f"Generating bundled starter corpus at {starter}…")
    run([sys.executable, str(generator)], cwd=REPO_ROOT)


def frontend_dist_ready() -> bool:
    return FRONTEND_DIST.is_dir() and (FRONTEND_DIST / "index.html").is_file()


def _frontend_watch_paths() -> tuple[Path, ...]:
    frontend_dir = REPO_ROOT / "frontend"
    return (
        frontend_dir / "src",
        frontend_dir / "index.html",
        frontend_dir / "package.json",
        frontend_dir / "package-lock.json",
        frontend_dir / "vite.config.ts",
        frontend_dir / "tsconfig.json",
    )


def compute_frontend_src_fingerprint() -> str:
    """Stable hash of UI inputs (git-friendly; avoids false stale warnings from mtimes on CAI)."""
    import hashlib

    digest = hashlib.sha256()
    for root in _frontend_watch_paths():
        if not root.exists():
            continue
        paths = [root] if root.is_file() else sorted(root.rglob("*"))
        for path in paths:
            if not path.is_file():
                continue
            if path.suffix in {".map"}:
                continue
            rel = path.relative_to(REPO_ROOT).as_posix()
            digest.update(rel.encode("utf-8"))
            digest.update(str(path.stat().st_size).encode("utf-8"))
            with path.open("rb") as fp:
                while chunk := fp.read(65536):
                    digest.update(chunk)
    return digest.hexdigest()[:24]


def _read_stamp_fingerprint() -> str | None:
    stamp = FRONTEND_DIST / ".ui-build-stamp"
    if not stamp.is_file():
        return None
    for line in stamp.read_text(encoding="utf-8").splitlines():
        if line.startswith("src_fingerprint="):
            return line.split("=", 1)[1].strip() or None
    return None


def frontend_dist_stale() -> bool:
    """True when UI sources changed since the last stamped production build."""
    if not frontend_dist_ready():
        return True
    current = compute_frontend_src_fingerprint()
    stamped = _read_stamp_fingerprint()
    if stamped:
        return stamped != current
    # Legacy bundles without fingerprint: fall back to mtime (may false-positive on CAI).
    dist_mtime = (FRONTEND_DIST / "index.html").stat().st_mtime
    for root in _frontend_watch_paths():
        if not root.exists():
            continue
        if root.is_file():
            if root.stat().st_mtime > dist_mtime:
                return True
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.stat().st_mtime > dist_mtime:
                return True
    return False


def read_ui_build_label() -> str | None:
    stamp = FRONTEND_DIST / ".ui-build-stamp"
    if stamp.is_file():
        return stamp.read_text(encoding="utf-8").strip() or None
    return None


def read_platform_version() -> str:
    version_file = REPO_ROOT / "VERSION"
    if not version_file.is_file():
        return "1.1.0.0"
    import re

    pat = re.compile(r"^(\d+\.\d+\.\d+\.\d+)\s*$")
    for raw in version_file.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if pat.match(line):
            return line
    return "1.1.0.0"


def write_ui_build_stamp() -> None:
    if not frontend_dist_ready():
        return
    stamp = FRONTEND_DIST / ".ui-build-stamp"
    js_name = ""
    assets = FRONTEND_DIST / "assets"
    if assets.is_dir():
        for js_path in sorted(assets.glob("index-*.js")):
            js_name = js_path.name
            break
    stamp.write_text(
        f"platform_version={read_platform_version()}\n"
        f"built_at={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n"
        f"src_fingerprint={compute_frontend_src_fingerprint()}\n"
        f"bundle_js={js_name}\n",
        encoding="utf-8",
    )


def ui_bundle_has_sidebar_ux() -> bool:
    """Heuristic: current production bundle includes the sidebar shell."""
    assets = FRONTEND_DIST / "assets"
    if not assets.is_dir():
        return False
    for js_path in assets.glob("index-*.js"):
        try:
            text = js_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if (
            "app-shell" in text
            and "Framework Configuration" in text
            and "model-tasks-recipe-layout" in text
        ):
            return True
    return False


def ui_bundle_runtime_sanity() -> tuple[bool, str]:
    """
    Catch shipped bundles that load but crash React at runtime (e.g. ReferenceError).
    """
    import re

    assets = FRONTEND_DIST / "assets"
    index = FRONTEND_DIST / "index.html"
    if not index.is_file() or not assets.is_dir():
        return False, "frontend/dist is incomplete"
    html = index.read_text(encoding="utf-8", errors="ignore")
    js_match = re.search(r'(?:\./|/)assets/(index-[^"\']+\.js)', html)
    if not js_match:
        return False, "index.html is missing a ./assets/index-*.js script tag"
    js_name = js_match.group(1)
    js_path = assets / js_name
    if not js_path.is_file():
        return (
            False,
            f"index.html references {js_name} but that file is missing — run scripts/build_ui.py",
        )
    text = js_path.read_text(encoding="utf-8", errors="ignore")
    if re.search(r"\buiBuildLabel\b", text):
        return (
            False,
            f"{js_name} is a broken UI bundle (uiBuildLabel ReferenceError). "
            "Run: python3 scripts/build_ui.py --force",
        )
    return True, ""


def ensure_prebuilt_ui_or_npm() -> tuple[bool, str]:
    """
    CAI/CDSW often has no Node. Return (ok, message) for whether UX can start.
    """
    if not frontend_dist_ready():
        if shutil.which("npm"):
            return False, "frontend/dist is missing; run: python3 scripts/build_ui.py"
        return (
            False,
            "frontend/dist is missing and npm is not installed. "
            "Run `git pull origin main` (dist is committed) or copy frontend/dist from a dev machine.",
        )
    sane, sanity_msg = ui_bundle_runtime_sanity()
    if not sane:
        return False, sanity_msg
    if ui_bundle_has_sidebar_ux():
        if frontend_dist_stale():
            return (
                True,
                "Using prebuilt frontend/dist (no npm on this machine). "
                "Sources differ from stamp — run `git pull origin main` for the latest "
                "committed bundle; `build_ui.py` cannot compile here without Node.",
            )
        return True, "Using prebuilt frontend/dist (no npm required)."
    if shutil.which("npm"):
        return False, "frontend/dist is present but outdated; run: python3 scripts/build_ui.py --force"
    return (
        False,
        "frontend/dist is outdated and npm is not available. "
        "Run `git pull origin main` to fetch the latest committed production bundle.",
    )


def cai_session_ip() -> str | None:
    return os.environ.get("CDSW_IP_ADDRESS") or os.environ.get("CML_IP_ADDRESS")


def is_cai_session() -> bool:
    return cai_session_ip() is not None


def api_health_url(port: str | int | None = None) -> str:
    p = port if port is not None else api_port()
    return f"http://{BIND_HOST}:{p}/api/health"


def ui_health_url(port: str | int) -> str:
    return f"http://{BIND_HOST}:{port}/"


def ui_api_health_url(port: str | int) -> str:
    """Health via the UX gateway (proves /api is proxied, not just static HTML)."""
    return f"http://{BIND_HOST}:{port}/api/health"


def ux_gateway_ready(port: str | int) -> bool:
    return http_ok(ui_api_health_url(port))


def resolve_ux_port() -> str:
    """CAI: CDSW_APP_PORT from the environment (read once at startup). Local: PLATFORM_UI_PORT."""
    if is_cai_session():
        port = os.environ.get("CDSW_APP_PORT")
        if not port:
            raise RuntimeError("CDSW_APP_PORT must be set in a CAI session.")
        return port
    return os.environ.get("PLATFORM_UI_PORT", "5173")


def _port_is_open(port: int) -> bool:
    import socket

    try:
        with socket.create_connection((BIND_HOST, port), timeout=1.0):
            return True
    except OSError:
        return False


def hold_forever() -> int:
    """Keep the Application process alive without binding CDSW_APP_PORT again."""
    print("Process will stay running (Ctrl+C to stop).", flush=True)
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        return 0


def run(
    cmd: Sequence[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(
        list(cmd),
        cwd=str(cwd or REPO_ROOT),
        env=merged,
        check=check,
        text=True,
    )


def tmux_available() -> bool:
    return shutil.which("tmux") is not None


def tmux_cmd(*args: str) -> list[str]:
    if Path(TMUX_CONF).is_file():
        return ["tmux", "-f", TMUX_CONF, *args]
    return ["tmux", *args]


def tmux_has_session(name: str) -> bool:
    if not tmux_available():
        return False
    return (
        subprocess.run(
            tmux_cmd("has-session", "-t", name),
            capture_output=True,
            text=True,
        ).returncode
        == 0
    )


def tmux_new_session(name: str, cwd: Path, shell_command: str) -> None:
    run(
        tmux_cmd(
            "new-session",
            "-d",
            "-s",
            name,
            "-c",
            str(cwd),
            shell_command,
        )
    )


def log_file(name: str) -> Path:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    return RUN_DIR / f"{name}.log"


def _pid_file(name: str) -> Path:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    return RUN_DIR / f"{name}.pid"


def process_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def clear_stale_pid(name: str) -> None:
    pid_path = _pid_file(name)
    if not pid_path.is_file():
        return
    try:
        pid = int(pid_path.read_text().strip())
    except ValueError:
        pid_path.unlink(missing_ok=True)
        return
    if not process_running(pid):
        pid_path.unlink(missing_ok=True)


def http_ok(url: str, timeout: float = 5.0) -> bool:
    try:
        with urlopen(url, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except (HTTPError, URLError, TimeoutError):
        return False


def wait_for_urls(urls: Sequence[str], attempts: int = 90, delay: float = 1.0) -> bool:
    for _ in range(attempts):
        if all(http_ok(u) for u in urls):
            return True
        time.sleep(delay)
    return False


def api_port() -> str:
    return os.environ.get("PLATFORM_API_PORT", "8000")


def platform_python() -> str:
    """Always use the active system interpreter (no project .venv)."""
    return os.environ.get("PLATFORM_PYTHON", sys.executable)


def venv_python() -> str:
    """Alias for platform_python (legacy name used by start/install scripts)."""
    return platform_python()


def venv_ray_argv(*ray_args: str) -> list[str]:
    py = platform_python()
    try:
        import ray  # noqa: F401
    except ImportError:
        raise FileNotFoundError(
            "Ray is not installed for system Python. Run: python3 scripts/install.py"
        ) from None
    return [py, "-m", "ray.scripts.scripts", *ray_args]


def venv_subprocess_env() -> dict[str, str]:
    merged = os.environ.copy()
    merged.pop("VIRTUAL_ENV", None)
    merged.setdefault("PIP_USER", "0")
    merged.pop("PYTHONNOUSERSITE", None)
    backend = str(BACKEND_DIR.resolve())
    path = merged.get("PYTHONPATH", "")
    merged["PYTHONPATH"] = f"{backend}{os.pathsep}{path}" if path else backend
    return merged


def pip_install(pip_args: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Install with CDSW-friendly fallbacks (PEP 668, --user)."""
    py = platform_python()
    if "--break-system-packages" in pip_args or "--user" in pip_args:
        return run([py, "-m", "pip", "install", *pip_args], env=venv_subprocess_env(), check=check)

    attempts: list[tuple[list[str], dict[str, str]]] = [
        ([py, "-m", "pip", "install", *pip_args], venv_subprocess_env()),
        (
            [py, "-m", "pip", "install", "--break-system-packages", *pip_args],
            venv_subprocess_env(),
        ),
    ]
    user_env = venv_subprocess_env()
    user_env["PIP_USER"] = "1"
    attempts.append(([py, "-m", "pip", "install", "--user", *pip_args], user_env))

    last_error: subprocess.CalledProcessError | None = None
    for cmd, env in attempts:
        try:
            return run(cmd, env=env, check=check)
        except subprocess.CalledProcessError as exc:
            last_error = exc
            print(f"pip attempt failed ({' '.join(cmd[4:6])}…): exit {exc.returncode}", file=sys.stderr)
    if last_error:
        raise last_error
    raise RuntimeError("pip install failed with no attempts")


def python_can_import(*modules: str) -> bool:
    """True when platform_python() can import modules (always checked in a subprocess)."""
    py = platform_python()
    script = "import " + ", ".join(modules)
    result = subprocess.run(
        [py, "-c", script],
        cwd=str(REPO_ROOT),
        env=venv_subprocess_env(),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 and result.stderr:
        print(result.stderr.strip(), file=sys.stderr)
    return result.returncode == 0


def uvicorn_cli_ready() -> bool:
    py = platform_python()
    result = subprocess.run(
        [py, "-m", "uvicorn", "--version"],
        cwd=str(REPO_ROOT),
        env=venv_subprocess_env(),
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def install_minimal_runtime() -> None:
    """Fast path so API/UX can start even if full requirements.txt is heavy."""
    print("Installing minimal API runtime (uvicorn, fastapi, sqlalchemy)…", flush=True)
    pip_install(
        [
            "uvicorn[standard]>=0.32.0",
            "fastapi>=0.115.0",
            "numpy>=2.0.0,<3",
            "sqlalchemy>=2.0.36",
            "pydantic>=2.9.0",
            "pydantic-settings>=2.6.0",
            "httpx>=0.27.0",
            "python-multipart>=0.0.12",
        ]
    )


def ensure_runtime_ready() -> int:
    """
    Ensure system Python has packages needed to run API + UX (uvicorn, FastAPI, …).
    Runs pip automatically when imports are missing.
    """
    if os.environ.get("PLATFORM_SKIP_DEP_CHECK", "").lower() in ("1", "true", "yes"):
        return 0

    py = platform_python()
    if python_can_import(*RUNTIME_IMPORTS) and uvicorn_cli_ready():
        print(f"Python runtime ready: {py}")
        return 0

    print(f"Missing packages for {py} — installing dependencies…", flush=True)
    from install import install_python_deps  # noqa: PLC0415

    try:
        install_python_deps()
    except subprocess.CalledProcessError:
        print("Full requirements install failed; trying minimal runtime packages.", file=sys.stderr)
        try:
            install_minimal_runtime()
        except subprocess.CalledProcessError as exc:
            print(f"Minimal install failed (exit {exc.returncode}).", file=sys.stderr)
            return 1

    if not uvicorn_cli_ready():
        print(
            "uvicorn still not available. Run manually:\n"
            f"  {py} -m pip install 'uvicorn[standard]>=0.32.0'\n"
            f"  {py} -m pip install -r backend/requirements.txt",
            file=sys.stderr,
        )
        return 1

    if not python_can_import(*RUNTIME_IMPORTS):
        print("Warning: some runtime imports still missing; API may be limited.", file=sys.stderr)

    print("Python dependencies ready.")
    return 0


def _ray_port_open(port: int = 6379) -> bool:
    import socket

    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1.0):
            return True
    except OSError:
        return False


def print_cai_memory_hints() -> None:
    if not is_cai_session():
        return
    backend = REPO_ROOT / "backend"
    if str(backend) not in sys.path:
        sys.path.insert(0, str(backend))
    from app.resource_hints import (  # noqa: PLC0415
        RECOMMENDED_SESSION_MEMORY_GB,
        RECOMMENDED_WORKER_MEMORY_GB,
        container_memory_limit_gb,
        is_low_memory_container,
    )

    limit = container_memory_limit_gb()
    if limit is not None:
        print(
            f"CAI container memory limit ~{limit:.2f} GiB "
            f"(recommended session >={RECOMMENDED_SESSION_MEMORY_GB:.0f} GiB)."
        )
    if is_low_memory_container():
        print(
            "Low-memory mode: Ray Train uses 1 worker and batch<=2. "
            "Set a larger CML resource profile or PLATFORM_CAI_LOW_MEMORY=0 "
            "when using >=8 GiB workers.",
            file=sys.stderr,
        )
    print(
        f"CDSW workers: set CML_RESOURCE_PROFILE_ID to >={RECOMMENDED_WORKER_MEMORY_GB:.0f} GiB "
        f"(DEFAULT_WORKER_MEMORY_GB env, default 8 — platform minimum)."
    )


def _ray_head_options():
    backend = REPO_ROOT / "backend"
    if str(backend) not in sys.path:
        sys.path.insert(0, str(backend))
    from app.services.ray_head_options import (
        ray_start_include_dashboard,
        state_api_reachable,
    )

    return ray_start_include_dashboard, state_api_reachable


def start_ray_head() -> None:
    ip = cai_session_ip()
    if not ip:
        return
    if os.environ.get("RAY_SKIP_HEAD", "").lower() in ("1", "true", "yes"):
        print("Skipping Ray head startup (RAY_SKIP_HEAD is set).")
        return

    os.environ["RAY_HEAD_ADDRESS"] = f"{ip}:6379"
    include_dashboard_fn, state_api_reachable = _ray_head_options()

    if _ray_port_open(6379):
        print("Ray head already listening on :6379")
        os.environ["RAY_ADDRESS"] = "127.0.0.1:6379"
        if not state_api_reachable():
            print(
                "WARNING: Ray GCS is up but the API server (State API) is not reachable on "
                ":8265. Ray Train may log PlacementGroupCleaner warnings until you "
                "`ray stop` and restart this app (dashboard starts automatically when "
                "ray[default] is installed). Or set RAY_INCLUDE_DASHBOARD=1 before start.",
                file=sys.stderr,
            )
        return

    include_dashboard = include_dashboard_fn()
    ray_args = ["start", "--head", "--port=6379"]
    backend = REPO_ROOT / "backend"
    if str(backend) not in sys.path:
        sys.path.insert(0, str(backend))
    from app.resource_hints import container_memory_limit_gb, ray_head_start_memory_args  # noqa: PLC0415

    mem_args = ray_head_start_memory_args()
    ray_args.extend(mem_args)
    if mem_args:
        limit = container_memory_limit_gb()
        print(
            f"Minimum-memory Ray profile (~{limit:.1f} GiB session): "
            + " ".join(mem_args),
        )
    if include_dashboard:
        ray_args.extend(["--dashboard-port=8265", "--include-dashboard=true"])
    else:
        ray_args.append("--include-dashboard=false")

    print(f"CAI session at {ip}; starting Ray head (dashboard={include_dashboard})...")
    try:
        result = run(
            venv_ray_argv(*ray_args),
            env=venv_subprocess_env(),
            check=False,
        )
    except FileNotFoundError as exc:
        print(f"Warning: {exc}", file=sys.stderr)
        return

    if result.returncode != 0:
        print(
            "Warning: Ray head did not start. Training may use local Ray or workers "
            "once the cluster is up. To skip this step: RAY_SKIP_HEAD=1. "
            "Install ray[default] (see backend/requirements.txt) or set RAY_INCLUDE_DASHBOARD=0 "
            "only if you accept Ray Train State API warnings.",
            file=sys.stderr,
        )
        return
    os.environ["RAY_ADDRESS"] = "127.0.0.1:6379"


def start_background_process(
    name: str,
    *,
    cwd: Path,
    argv: Sequence[str],
    env: dict[str, str] | None = None,
) -> None:
    log_path = log_file(name)
    merged = os.environ.copy()
    if env:
        merged.update(env)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab", buffering=0) as log_fp:
        proc = subprocess.Popen(
            list(argv),
            cwd=str(cwd),
            env=merged,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    _pid_file(name).write_text(str(proc.pid))
    print(f"Started {name} pid={proc.pid} log={log_path}")


def _api_upstream(api: str | int) -> str:
    return f"http://127.0.0.1:{api}"


def _ui_gateway_env(api: str | int) -> dict[str, str]:
    env = venv_subprocess_env()
    env["PLATFORM_API_UPSTREAM"] = _api_upstream(api)
    return env


def _uvicorn_argv(app_target: str, port: str | int) -> list[str]:
    if ensure_runtime_ready() != 0:
        raise RuntimeError(
            f"uvicorn is not installed for {platform_python()}. "
            "Run: python3 scripts/install.py"
        )
    argv = [
        platform_python(),
        "-m",
        "uvicorn",
        app_target,
        "--host",
        BIND_HOST,
        "--port",
        str(port),
    ]
    verbose_access = os.environ.get("PLATFORM_ACCESS_LOG", "").lower() in (
        "1",
        "true",
        "yes",
    )
    if verbose_access:
        log_level = os.environ.get("PLATFORM_UVICORN_LOG_LEVEL", "info").strip().lower()
        argv.extend(["--log-level", log_level])
    else:
        log_config = BACKEND_DIR / "uvicorn_log_config.json"
        if log_config.is_file():
            argv.extend(["--log-config", str(log_config.resolve())])
        else:
            argv.extend(["--log-level", "error", "--no-access-log"])
    return argv


def run_ui_gateway_foreground(ui: str | int, api: str | int) -> int:
    """Bind UX exactly once on CDSW_APP_PORT (127.0.0.1 only)."""
    ui_s = str(ui)
    if ux_gateway_ready(ui_s):
        print(
            f"UX gateway already serving /api on {BIND_HOST}:{ui_s} (CDSW_APP_PORT); "
            "not starting a second listener."
        )
        return hold_forever()
    if _port_is_open(int(ui_s)):
        print(
            f"Port {ui_s} (CDSW_APP_PORT) is in use but /api/health is not proxied. "
            f"Stop the other listener or restart the session, then run start.py again.",
            file=sys.stderr,
        )
        if http_ok(api_health_url(api)):
            print(
                f"API is up on {BIND_HOST}:{api}; the Application port must run app.ui_gateway.",
                file=sys.stderr,
            )
        return 1

    print(
        f"Starting UX on {BIND_HOST}:{ui_s} (CDSW_APP_PORT; API on {BIND_HOST}:{api}) ..."
    )
    return subprocess.call(
        _uvicorn_argv("app.ui_gateway:app", ui_s),
        cwd=str(BACKEND_DIR),
        env=_ui_gateway_env(api),
    )


def start_ui_gateway_background(ui: str | int, api: str | int) -> None:
    ui_s = str(ui)
    if ux_gateway_ready(ui_s):
        print(f"UX gateway already serving /api on {BIND_HOST}:{ui_s}")
        return
    if _port_is_open(int(ui_s)):
        print(
            f"Port {ui_s} in use without /api proxy; not starting UX.",
            file=sys.stderr,
        )
        return
    print(f"Starting UX on {BIND_HOST}:{ui_s} ...")
    clear_stale_pid("ui")
    start_background_process(
        "ui",
        cwd=BACKEND_DIR,
        argv=_uvicorn_argv("app.ui_gateway:app", ui_s),
        env=_ui_gateway_env(api),
    )


def ensure_api_running(port: str | int | None = None) -> None:
    if ensure_runtime_ready() != 0:
        raise RuntimeError("Cannot start API: Python dependencies are not installed.")
    port = port if port is not None else api_port()
    health = api_health_url(port)
    if http_ok(health):
        if uvicorn_cli_ready():
            print(f"API already responding on :{port}")
            return
        print(
            f"Port :{port} responds but uvicorn is missing for {platform_python()}; "
            "starting a new API process after install.",
            file=sys.stderr,
        )
    print(f"Starting API on :{port} ...")
    py = platform_python()
    uvicorn_argv = _uvicorn_argv("app.main:app", port)
    if tmux_available():
        if not tmux_has_session("api"):
            api_cmd = (
                f'"{py}" -m uvicorn app.main:app --host {BIND_HOST} --port {port} '
                f"2>&1 | tee {log_file('api')}"
            )
            tmux_new_session("api", BACKEND_DIR, api_cmd)
        return
    clear_stale_pid("api")
    start_background_process(
        "api",
        cwd=BACKEND_DIR,
        argv=uvicorn_argv,
        env=venv_subprocess_env(),
    )


def run_vite_dev_foreground(port: str) -> int:
    """Optional hot-reload dev server (requires npm). Binds CDSW_APP_PORT once."""
    npm = shutil.which("npm")
    if not npm:
        raise RuntimeError("npm is required for --dev-ui (Vite dev server)")
    if http_ok(ui_api_health_url(port)) or _port_is_open(int(port)):
        print(f"Port {port} already in use; not starting Vite.", file=sys.stderr)
        return 1
    env = venv_subprocess_env()
    env["PLATFORM_UI_PORT"] = port
    print(f"Starting Vite dev server on {BIND_HOST}:{port} ...")
    return subprocess.call(
        [npm, "run", "dev"],
        cwd=str(REPO_ROOT / "frontend"),
        env=env,
    )
