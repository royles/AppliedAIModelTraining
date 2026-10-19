"""Default NCCL / PyTorch distributed env for GPU Ray Train on CAI worker networks."""

from __future__ import annotations

import os

# Applied to session driver, Ray Train workers, and CDSW worker bootstrap unless disabled.
NCCL_DEFAULTS: dict[str, str] = {
    # Fail fast on collective errors instead of hanging.
    "NCCL_ASYNC_ERROR_HANDLING": "1",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    # Prefer pod data plane (exclude loopback / docker bridge).
    "NCCL_SOCKET_IFNAME": "^docker0,lo",
    # Allow P2P across GPUs on the same host when available.
    "NCCL_P2P_DISABLE": "0",
}


def nccl_defaults_enabled() -> bool:
    return os.environ.get("PLATFORM_NCCL_DEFAULTS", "1").lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def default_nccl_env() -> dict[str, str]:
    if not nccl_defaults_enabled():
        return {}
    out = dict(NCCL_DEFAULTS)
    # Honour explicit NCCL_DEBUG from the environment only (no default spam).
    if os.environ.get("NCCL_DEBUG"):
        out["NCCL_DEBUG"] = os.environ["NCCL_DEBUG"]
    return out


def apply_nccl_env(target: dict[str, str] | None = None) -> dict[str, str]:
    """
    Merge NCCL defaults into `target` or os.environ without overriding user values.
    Returns the dict that was updated.
    """
    if target is None:
        for key, value in default_nccl_env().items():
            if value and not os.environ.get(key):
                os.environ[key] = value
        return dict(os.environ)

    for key, value in default_nccl_env().items():
        if value and not target.get(key):
            target[key] = value
    return target


def detect_collective_backend() -> str | None:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_backend()
    except Exception:  # noqa: BLE001
        return None
    return None
