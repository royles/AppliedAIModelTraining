"""Run Ray Train driver on a CDSW worker engine (not on the session head)."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from app.config import REPO_ROOT

logger = logging.getLogger(__name__)

REMOTE_PAYLOAD_NAME = "remote_train_payload.json"
REMOTE_RESULT_NAME = "remote_driver_result.json"


def remote_payload_path(artifact_dir: str) -> Path:
    return Path(artifact_dir) / REMOTE_PAYLOAD_NAME


def remote_result_path(artifact_dir: str) -> Path:
    return Path(artifact_dir) / REMOTE_RESULT_NAME


def write_remote_train_payload(
    artifact_dir: str,
    *,
    ray_address: str,
    num_workers: int,
    use_gpu: bool,
    train_config: dict[str, Any],
    ray_head_address: str | None = None,
    ray_auth_token: str | None = None,
) -> Path:
    path = remote_payload_path(artifact_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "ray_address": ray_address,
        "num_workers": num_workers,
        "use_gpu": use_gpu,
        "train_config": train_config,
    }
    if ray_head_address:
        payload["ray_head_address"] = ray_head_address
    if ray_auth_token:
        payload["ray_auth_token"] = ray_auth_token
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def poll_remote_driver_result(
    artifact_dir: str,
    *,
    timeout_sec: float = 6 * 3600.0,
    poll_sec: float = 10.0,
) -> dict[str, Any]:
    result_path = remote_result_path(artifact_dir)
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if result_path.is_file():
            try:
                data = json.loads(result_path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and data.get("status") in ("succeeded", "failed"):
                    return data
            except json.JSONDecodeError:
                pass
        time.sleep(poll_sec)
    raise TimeoutError(
        f"Remote Ray Train driver did not write {REMOTE_RESULT_NAME} within {timeout_sec:.0f}s"
    )


def build_remote_train_driver_code(job_id: int, head_address: str) -> str:
    """
    CDSW worker bootstrap: connect to Ray as a **client** on a remote engine, run Train.

    Does not run ``ray start --block`` (GPU workers already joined the cluster).
    """
    return f'''import json
import os
import sys
import traceback
from pathlib import Path

JOB_ID = {job_id}
FALLBACK_HEAD = "{head_address}"

def project_root() -> Path:
    for key in ("CDSW_PROJECT_ROOT", "PROJECT_ROOT", "CDSW_WORKING_DIRECTORY"):
        val = os.environ.get(key)
        if val:
            p = Path(val)
            if p.is_dir():
                return p.resolve()
    return Path.cwd().resolve()

def resolve_head() -> str:
    for key in ("CDSW_MASTER_IP", "CML_MASTER_IP"):
        ip = os.environ.get(key)
        if ip:
            return ip.strip() + ":6379"
    return FALLBACK_HEAD

root = project_root()
os.chdir(root)
sys.path.insert(0, str(root / "backend"))

artifact_dir = root / "data" / "models" / f"job_{{JOB_ID}}"
payload_path = artifact_dir / "{REMOTE_PAYLOAD_NAME}"
result_path = artifact_dir / "{REMOTE_RESULT_NAME}"

def write_result(status: str, **fields):
    artifact_dir.mkdir(parents=True, exist_ok=True)
    body = {{"status": status, **fields}}
    result_path.write_text(json.dumps(body), encoding="utf-8")

try:
    if not payload_path.is_file():
        raise FileNotFoundError(f"Missing {{payload_path}}")
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    head = resolve_head()
    node_ip = os.environ.get("CDSW_IP_ADDRESS") or os.environ.get("CML_IP_ADDRESS") or head.split(":")[0]
    os.environ["RAY_ADDRESS"] = head
    print(f"[platform] Remote Ray Train driver job={{JOB_ID}} head={{head}} node_ip={{node_ip}}", flush=True)

    token = (payload.get("ray_auth_token") or "").strip()
    if token:
        os.environ["RAY_AUTH_TOKEN"] = token
    else:
        token_path = Path.home() / ".ray" / "auth_token"
        if token_path.is_file():
            os.environ["RAY_AUTH_TOKEN"] = token_path.read_text(encoding="utf-8").strip()

    from app.services.ray_driver_connect import init_ray_train_driver

    worker_head = payload.get("ray_head_address") or head

    def _flog(msg: str) -> None:
        print(f"[platform] {{msg}}", flush=True)

    gcs = init_ray_train_driver(
        payload["ray_address"],
        worker_head=worker_head,
        session_only=False,
        framework_log=_flog,
    )
    print(f"[platform] Ray driver connected at {{gcs}}", flush=True)

    from app.training.transformer_job import run_transformer_training

    tc = dict(payload["train_config"])
    tc["remote_driver_engine"] = True
    if worker_head:
        tc["ray_head_address"] = worker_head
    result = run_transformer_training(
        ray_address=payload["ray_address"],
        num_workers=int(payload["num_workers"]),
        use_gpu=bool(payload["use_gpu"]),
        train_config=tc,
    )
    write_result("succeeded", metrics=result.get("metrics"), result=result)
    print("[platform] Remote Ray Train driver finished OK", flush=True)
except Exception as exc:
    write_result("failed", error=str(exc), traceback=traceback.format_exc())
    print(f"[platform] Remote Ray Train driver failed: {{exc}}", flush=True)
    raise
'''


def use_remote_train_driver(
    *,
    launch_cdsw_workers: bool,
    pool_is_remote: bool,
) -> bool:
    if not launch_cdsw_workers or not pool_is_remote:
        return False
    explicit = os.environ.get("PLATFORM_TRAIN_DRIVER_ON_SESSION", "").strip().lower()
    if explicit in ("1", "true", "yes"):
        return False
    return True
