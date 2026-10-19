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


def cuda_device_mapping_summary(device: object) -> str:
    """Human-readable confirmation that Ray/GPU maps to a CUDA device (for job logs)."""
    try:
        import torch

        if not isinstance(device, torch.device):
            device = torch.device(str(device))
    except Exception:  # noqa: BLE001
        return f"compute device={device!s} (CUDA status unknown)"

    if device.type != "cuda":
        cuda_ok = False
        try:
            import torch

            cuda_ok = torch.cuda.is_available()
        except Exception:  # noqa: BLE001
            pass
        return (
            f"compute mapped to {device} (torch.cuda.is_available()={cuda_ok}; "
            "training on CPU)"
        )

    import torch

    idx = device.index
    if idx is None:
        idx = torch.cuda.current_device()
    try:
        name = torch.cuda.get_device_name(idx)
    except Exception:  # noqa: BLE001
        name = "unknown"
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)")
    return (
        f"GPU mapped to CUDA cuda:{idx} ({name}); "
        f"torch.cuda.is_available()=True; CUDA_VISIBLE_DEVICES={cvd}"
    )


def nccl_networking_summary(
    *,
    world_size: int = 1,
    collective_backend: str | None = None,
) -> str:
    """Confirm platform NCCL env and socket interface selection (for job logs)."""
    if not nccl_defaults_enabled():
        return (
            "NCCL platform defaults off (PLATFORM_NCCL_DEFAULTS=0); "
            "using only explicit NCCL_* / TORCH_NCCL_* env"
        )

    ifname = os.environ.get("NCCL_SOCKET_IFNAME", NCCL_DEFAULTS["NCCL_SOCKET_IFNAME"])
    async_nccl = os.environ.get(
        "NCCL_ASYNC_ERROR_HANDLING", NCCL_DEFAULTS["NCCL_ASYNC_ERROR_HANDLING"]
    )
    async_torch = os.environ.get(
        "TORCH_NCCL_ASYNC_ERROR_HANDLING",
        NCCL_DEFAULTS["TORCH_NCCL_ASYNC_ERROR_HANDLING"],
    )
    p2p = os.environ.get("NCCL_P2P_DISABLE", NCCL_DEFAULTS["NCCL_P2P_DISABLE"])

    parts = [
        "NCCL enabled for GPU networking",
        f"socket_ifname={ifname}",
        f"async_error_handling NCCL={async_nccl} TORCH={async_torch}",
        f"p2p_disable={p2p}",
    ]
    if collective_backend:
        parts.append(f"collective_backend={collective_backend}")
    elif world_size > 1:
        parts.append("collective_backend=nccl (expected for CUDA multi-rank)")
    return "; ".join(parts)
