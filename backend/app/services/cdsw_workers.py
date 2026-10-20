"""
Cloudera AI / CDSW workers integration for Ray GPU clusters.

Uses the workers API documented at:
https://docs.cloudera.com/machine-learning/cloud/distributed-computing/topics/ml-workers-api.html

Supports both legacy `cdsw` and `cml.workers_v1` entry points when running inside
Cloudera Machine Learning. Outside CML, operations are simulated for local development.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Protocol

from app.config import REPO_ROOT
from app.resource_hints import log_worker_launch_memory
from app.services.cai_networking import ray_driver_gcs_address, ray_worker_head_address
from app.services.nccl_env import apply_nccl_env

logger = logging.getLogger(__name__)


class WorkersClient(Protocol):
    def launch_workers(
        self,
        n: int,
        cpu: float,
        memory: float,
        *,
        nvidia_gpu: int = 0,
        code: str = "",
        env: dict[str, str] | None = None,
        accelerator_label_id: int | None = None,
        resource_profile_id: int | None = None,
        resource_group_id: int | None = None,
    ) -> Any: ...

    def stop_workers(self, *worker_ids: int) -> None: ...

    def list_workers(self) -> list[dict[str, Any]]: ...


@dataclass
class LaunchedWorkers:
    worker_ids: list[str]
    raw: Any


def _import_workers_client() -> WorkersClient | None:
    try:
        import cml.workers_v1 as workers  # type: ignore[import-untyped]

        return workers
    except ImportError:
        pass
    try:
        import cdsw.workers as workers  # type: ignore[import-untyped]

        return workers
    except ImportError:
        pass
    try:
        import cdsw  # type: ignore[import-untyped]

        if hasattr(cdsw, "launch_workers"):
            return cdsw
    except ImportError:
        pass
    return None


def cdsw_available() -> bool:
    return _import_workers_client() is not None


def ray_head_address() -> str | None:
    """Worker-facing Ray head (master CDSW_IP_ADDRESS:6379)."""
    return ray_worker_head_address()


def ray_client_address(head: str) -> str:
    host = head.split(":")[0]
    return f"ray://{host}:10001"


def _local_gcs_reachable(port: int = 6379) -> bool:
    import socket

    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1.0):
            return True
    except OSError:
        return False


def ray_driver_init_address(head: str | None) -> str | None:
    """GCS address for the Train driver (see docs/CAI_RAY_NETWORKING.md)."""
    from app.services.ray_driver_connect import resolve_driver_gcs_address

    if head:
        return resolve_driver_gcs_address(head, worker_head=head)
    env = os.environ.get("RAY_ADDRESS")
    resolved = resolve_driver_gcs_address(env)
    if resolved:
        return resolved
    if cdsw_available():
        from app.services.cai_networking import ray_driver_gcs_address

        return ray_driver_gcs_address(os.environ.get("RAY_HEAD_ADDRESS"))
    return env


def _parse_gcs_port(address: str, default: int = 6379) -> int:
    if ":" not in address:
        return default
    try:
        return int(address.rsplit(":", 1)[-1])
    except ValueError:
        return default


def wait_for_gcs_port(
    host: str,
    port: int = 6379,
    *,
    timeout_sec: float = 180.0,
    poll_sec: float = 2.0,
) -> bool:
    """Block until Ray GCS accepts TCP connections (returns False on timeout)."""
    import socket
    import time

    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2.0):
                return True
        except OSError:
            time.sleep(poll_sec)
    return False


def build_worker_start_code(head_address: str, *, cpus: float = 4.0, gpus: int = 1) -> str:
    """
    Python code for launch_workers `code=`.

    Always uses sys.executable (system Python). Installs backend/requirements.txt
    when Ray is missing (worker engines start with a bare interpreter).
    """
    # head_address is session-controlled (IP:port); no user strings in code path.
    cpu_i = max(1, int(cpus))
    gpu_i = max(0, int(gpus))
    return f'''import os
import subprocess
import sys
from pathlib import Path

FALLBACK_HEAD = "{head_address}"
WORKER_CPUS = {cpu_i}
WORKER_GPUS = {gpu_i}

def project_root() -> Path:
    for key in ("CDSW_PROJECT_ROOT", "PROJECT_ROOT", "CDSW_WORKING_DIRECTORY"):
        val = os.environ.get(key)
        if val:
            p = Path(val)
            if p.is_dir():
                return p.resolve()
    return Path.cwd().resolve()

def ray_importable(py: str) -> bool:
    return (
        subprocess.call(
            [py, "-c", "import ray"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        == 0
    )

def ensure_ray(py: str, root: Path) -> None:
    if ray_importable(py):
        return
    req = root / "backend" / "requirements.txt"
    env = {{**os.environ, "PIP_USER": "0"}}
    if req.is_file():
        subprocess.check_call([py, "-m", "pip", "install", "-q", "-r", str(req)], env=env, cwd=str(root))
    else:
        subprocess.check_call([py, "-m", "pip", "install", "-q", "ray"], env=env)

def pick_python(root: Path) -> str:
    venv_py = root / ".venv" / "bin" / "python"
    candidates: list[str] = []
    if venv_py.is_file():
        candidates.append(str(venv_py))
    if sys.executable not in candidates:
        candidates.append(sys.executable)
    for candidate in candidates:
        if ray_importable(candidate):
            return candidate
    return candidates[0]

root = project_root()
os.chdir(root)
py = pick_python(root)
ensure_ray(py, root)
if not ray_importable(py):
    alt = sys.executable
    if alt != py and ray_importable(alt):
        print(f"[platform] WARN: {{py}} has no ray; using {{alt}}", flush=True)
        py = alt
    else:
        print(f"[platform] ERROR: cannot import ray in {{py}} (or {{alt}})", flush=True)
        raise SystemExit(1)

def worker_node_ip() -> str:
    for key in ("CDSW_IP_ADDRESS", "CML_IP_ADDRESS"):
        ip = os.environ.get(key)
        if ip:
            return ip.strip()
    return ""

def resolve_head() -> str:
    pinned = os.environ.get("PLATFORM_RAY_HEAD_ADDRESS", "").strip()
    if pinned:
        return pinned
    for key in ("CDSW_MASTER_IP", "CML_MASTER_IP"):
        ip = os.environ.get(key)
        if ip:
            return ip.strip() + ":6379"
    return FALLBACK_HEAD

def wait_for_gcs(host: str, port: int, timeout_sec: float = 600.0) -> bool:
    import socket
    import time
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=3.0):
                return True
        except OSError:
            time.sleep(2.0)
    return False

HEAD = resolve_head()
host, _, port_s = HEAD.partition(":")
port = int(port_s or "6379")
print(f"[platform] Waiting for Ray GCS at {{HEAD}} (up to 600s)…", flush=True)
if not wait_for_gcs(host, port):
    print(f"[platform] ERROR: GCS not reachable at {{HEAD}}", flush=True)
    raise SystemExit(1)
NODE_IP = worker_node_ip()
print(
    f"[platform] CDSW worker python={{py}} head={{HEAD}} node_ip={{NODE_IP or 'auto'}} "
    f"cpus={{WORKER_CPUS}} gpus={{WORKER_GPUS}}",
    flush=True,
)
cmd = [
    py, "-m", "ray.scripts.scripts", "start", "--block",
    f"--address={{HEAD}}",
    f"--num-cpus={{WORKER_CPUS}}",
    f"--num-gpus={{WORKER_GPUS}}",
]
obj = os.environ.get("PLATFORM_RAY_OBJECT_STORE_BYTES")
if obj:
    cmd.append(f"--object-store-memory={{obj}}")
node_mem = os.environ.get("PLATFORM_RAY_NODE_MEMORY_BYTES")
if node_mem:
    cmd.append(f"--memory={{node_mem}}")
if NODE_IP:
    cmd.append(f"--node-ip-address={{NODE_IP}}")
print(f"[platform] exec: {{' '.join(cmd)}}", flush=True)
rc = subprocess.call(cmd)
print(f"[platform] ray start --block exited with code {{rc}}", flush=True)
print("[platform] see ray_process_exit.log in the Ray session dir on this engine", flush=True)
raise SystemExit(rc)
'''


def worker_launch_env(
    project_root: Path | None = None,
    *,
    cpus: float | None = None,
    gpus: int | None = None,
    memory_gb: float | None = None,
    ray_head_address: str | None = None,
) -> dict[str, str]:
    root = (project_root or REPO_ROOT).resolve()
    env: dict[str, str] = {
        "PIP_USER": "0",
        "CDSW_PROJECT_ROOT": str(root),
        "PROJECT_ROOT": str(root),
    }
    if cpus is not None:
        env["PLATFORM_WORKER_CPU_COUNT"] = str(int(max(1, cpus)))
    if gpus is not None:
        env["PLATFORM_WORKER_GPU_COUNT"] = str(max(0, int(gpus)))
    apply_nccl_env(env)
    from app.services.ray_memory_config import worker_ray_memory_env

    mem_gb = memory_gb
    if mem_gb is None:
        mem_gb = float(os.environ.get("PLATFORM_WORKER_MEMORY_GB", "8") or 8)
    env["PLATFORM_WORKER_MEMORY_GB"] = str(mem_gb)
    env.update(worker_ray_memory_env(mem_gb))
    from app.services.cai_networking import master_engine_ip, parse_gcs_host_port

    head = (ray_head_address or "").strip()
    if head:
        env["PLATFORM_RAY_HEAD_ADDRESS"] = head
        host, _ = parse_gcs_host_port(head)
        if host:
            env["CDSW_MASTER_IP"] = host
            env["CML_MASTER_IP"] = host
    else:
        master = master_engine_ip()
        if master:
            env["CDSW_MASTER_IP"] = master
            env["CML_MASTER_IP"] = master
    for key in (
        "CDSW_WORKING_DIRECTORY",
        "HOME",
        "PATH",
        "PYTHONPATH",
        "CML_RESOURCE_PROFILE_ID",
        "RAY_AUTH_TOKEN",
        "RAY_AUTH_TOKEN_PATH",
    ):
        if os.environ.get(key):
            env[key] = os.environ[key]
    return env


class CdswRayWorkerPool:
    """Provision GPU workers via cdsw/cml workers API and attach them to Ray."""

    def __init__(
        self,
        cpu: float = 4.0,
        memory_gb: float = 8.0,
        gpus_per_worker: int = 1,
        *,
        accelerator_label_id: str | None = None,
        resource_profile_id: str | None = None,
        resource_group_id: str | None = None,
    ) -> None:
        self.cpu = cpu
        self.memory_gb = memory_gb
        self.gpus_per_worker = gpus_per_worker
        self._accelerator_label_id = accelerator_label_id
        self._resource_profile_id = resource_profile_id
        self._resource_group_id = resource_group_id
        self._client = _import_workers_client()
        self._launched: LaunchedWorkers | None = None

    @property
    def is_remote(self) -> bool:
        return self._client is not None

    def launch(
        self,
        count: int,
        head_address: str,
        *,
        code: str | None = None,
        cpu: float | None = None,
        memory_gb: float | None = None,
        gpus_per_worker: int | None = None,
    ) -> LaunchedWorkers:
        if count < 1:
            raise ValueError("count must be >= 1")
        worker_cpu = cpu if cpu is not None else self.cpu
        worker_mem = memory_gb if memory_gb is not None else self.memory_gb
        worker_gpus = gpus_per_worker if gpus_per_worker is not None else self.gpus_per_worker
        code = code or build_worker_start_code(
            head_address, cpus=worker_cpu, gpus=worker_gpus
        )
        if self._client is None:
            simulated = [f"local-sim-{i}" for i in range(count)]
            logger.info(
                "CDSW/CML workers API not available; simulating %s workers", count
            )
            self._launched = LaunchedWorkers(worker_ids=simulated, raw=simulated)
            return self._launched

        env = worker_launch_env(
            cpus=worker_cpu,
            gpus=worker_gpus,
            memory_gb=worker_mem,
            ray_head_address=head_address,
        )
        for key in ("RAY_AUTH_TOKEN", "RAY_AUTH_TOKEN_PATH", "RAY_ADDRESS"):
            if os.environ.get(key):
                env[key] = os.environ[key]
        kwargs: dict[str, Any] = {
            "n": count,
            "cpu": worker_cpu,
            "memory": worker_mem,
            "nvidia_gpu": worker_gpus,
            "code": code,
            "env": env,
        }
        optional = {
            "accelerator_label_id": self._accelerator_label_id
            or os.environ.get("CML_ACCELERATOR_LABEL_ID"),
            "resource_profile_id": self._resource_profile_id
            or os.environ.get("CML_RESOURCE_PROFILE_ID"),
            "resource_group_id": self._resource_group_id
            or os.environ.get("CML_RESOURCE_GROUP_ID"),
        }
        for key, val in optional.items():
            if val is not None and str(val).strip():
                kwargs[key] = int(val)

        log_worker_launch_memory(worker_mem, worker_gpus)
        raw = self._client.launch_workers(**kwargs)
        worker_ids = _extract_worker_ids(raw)
        if self._launched and self._launched.worker_ids:
            worker_ids = self._launched.worker_ids + worker_ids
        self._launched = LaunchedWorkers(worker_ids=worker_ids, raw=raw)
        logger.info("Launched %s CDSW workers for Ray head %s", count, head_address)
        return LaunchedWorkers(worker_ids=worker_ids[-count:], raw=raw)

    def list_workers_safe(self) -> list[dict[str, Any]]:
        if self._client is None:
            return []
        try:
            return self._client.list_workers()
        except Exception:  # noqa: BLE001
            logger.exception("list_workers failed")
            return []

    def scale(self, target: int, head_address: str) -> LaunchedWorkers:
        current = len(self._launched.worker_ids) if self._launched else 0
        if target > current:
            delta = target - current
            extra = self.launch(delta, head_address)
            merged = (self._launched.worker_ids if self._launched else []) + extra.worker_ids
            self._launched = LaunchedWorkers(worker_ids=merged, raw=extra.raw)
        elif target < current and self._launched:
            to_stop = self._launched.worker_ids[target:]
            self.stop_ids(to_stop)
            self._launched = LaunchedWorkers(
                worker_ids=self._launched.worker_ids[:target], raw=self._launched.raw
            )
        return self._launched or LaunchedWorkers(worker_ids=[], raw=[])

    def stop_ids(self, worker_ids: list[str]) -> None:
        if not worker_ids or self._client is None:
            return
        numeric = [int(w) for w in worker_ids if str(w).isdigit()]
        if numeric:
            self._client.stop_workers(*numeric)

    def stop_all(self) -> None:
        if self._launched:
            self.stop_ids(self._launched.worker_ids)
            self._launched = None
        if self._client is not None:
            try:
                self._client.stop_workers()
            except Exception:  # noqa: BLE001
                logger.exception("Failed to stop all workers")


def _extract_worker_ids(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        ids: list[str] = []
        for item in raw:
            if isinstance(item, dict) and "id" in item:
                ids.append(str(item["id"]))
            else:
                ids.append(str(item))
        return ids
    if isinstance(raw, dict) and "id" in raw:
        return [str(raw["id"])]
    return [str(raw)]


def worker_ids_to_json(ids: list[str]) -> str:
    return json.dumps(ids)
