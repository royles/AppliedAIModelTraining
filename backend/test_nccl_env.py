"""NCCL default env helpers."""

import os

from app.services.nccl_env import (
    apply_nccl_env,
    cuda_device_mapping_summary,
    default_nccl_env,
    nccl_defaults_enabled,
    nccl_networking_summary,
)


def test_nccl_defaults_on_by_default(monkeypatch) -> None:
    monkeypatch.delenv("PLATFORM_NCCL_DEFAULTS", raising=False)
    assert nccl_defaults_enabled() is True
    env = default_nccl_env()
    assert env["NCCL_ASYNC_ERROR_HANDLING"] == "1"
    assert env["NCCL_SOCKET_IFNAME"] == "^docker0,lo"


def test_nccl_defaults_can_disable(monkeypatch) -> None:
    monkeypatch.setenv("PLATFORM_NCCL_DEFAULTS", "0")
    assert nccl_defaults_enabled() is False
    assert default_nccl_env() == {}


def test_apply_nccl_env_does_not_override_user(monkeypatch) -> None:
    monkeypatch.delenv("PLATFORM_NCCL_DEFAULTS", raising=False)
    target = {"NCCL_ASYNC_ERROR_HANDLING": "0"}
    apply_nccl_env(target)
    assert target["NCCL_ASYNC_ERROR_HANDLING"] == "0"
    assert target.get("NCCL_P2P_DISABLE") == "0"


def test_apply_nccl_env_to_os_environ(monkeypatch) -> None:
    monkeypatch.delenv("PLATFORM_NCCL_DEFAULTS", raising=False)
    for key in ("NCCL_ASYNC_ERROR_HANDLING", "TORCH_NCCL_ASYNC_ERROR_HANDLING"):
        monkeypatch.delenv(key, raising=False)
    apply_nccl_env()
    assert os.environ.get("NCCL_ASYNC_ERROR_HANDLING") == "1"


def test_nccl_networking_summary_enabled(monkeypatch) -> None:
    monkeypatch.delenv("PLATFORM_NCCL_DEFAULTS", raising=False)
    text = nccl_networking_summary(world_size=2, collective_backend="nccl")
    assert "NCCL enabled" in text
    assert "socket_ifname=" in text
    assert "collective_backend=nccl" in text


def test_cuda_device_mapping_cpu() -> None:
    import torch

    text = cuda_device_mapping_summary(torch.device("cpu"))
    assert "cpu" in text.lower()
    assert "CUDA" in text or "cuda" in text
