"""ASCII / box-drawing prefixes for job logs (no emoji)."""

from __future__ import annotations

import math


def _bar(filled: int, width: int = 16) -> str:
    filled = max(0, min(width, filled))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def tag(kind: str) -> str:
    return {
        "queue": ">>",
        "worker": "=>",
        "master": "@@",
        "ray": "~~",
        "join": "<+",
        "train": "||",
        "step": "|>",
        "epoch": "E>",
        "warmup": "W>",
        "save": "S>",
        "done": "OK",
        "warn": "!!",
        "fail": "XX",
        "info": "..",
    }.get(kind, "..")


def framework_line(kind: str, message: str) -> str:
    return f"[{tag(kind)}] {message}"


def training_line(kind: str, message: str) -> str:
    return f"  {tag(kind)} {message}"


def worker_launch_line(index: int, total: int, worker_id: str, head: str) -> str:
    return framework_line(
        "worker",
        f"CDSW worker {index}/{total} id={worker_id} ──► Ray master {head}",
    )


def ray_node_joined_line(label: str) -> str:
    return framework_line("join", f"Ray node online ║ {label}")


def ray_cluster_ready_line(count: int) -> str:
    return framework_line("done", f"Cluster linked: {count} node(s) on master")


def train_worker_ready_line(
    rank: int,
    world_size: int,
    device: str,
    examples: int,
    max_steps: int,
) -> str:
    return training_line(
        "train",
        f"rank {rank + 1}/{world_size} ready device={device} "
        f"examples={examples} max_steps={max_steps}",
    )


def train_cuda_mapping_line(summary: str) -> str:
    return training_line("info", summary)


def train_nccl_network_line(summary: str) -> str:
    return training_line("info", summary)


def train_step_line(
    step: int,
    max_steps: int,
    loss: float,
    lr: float,
    epoch: int,
    steps_per_epoch: int,
) -> str:
    pct = int(100 * step / max(1, max_steps))
    bar = _bar(int(16 * step / max(1, max_steps)))
    ep_part = f"epoch~{epoch}" if steps_per_epoch > 0 else f"epoch={epoch}"
    return training_line(
        "step",
        f"{bar} {pct:3d}% step {step:4d}/{max_steps} loss={loss:.4f} "
        f"lr={lr:.2e} {ep_part}",
    )


def train_epoch_start_line(epoch: int) -> str:
    return training_line("epoch", f"── dataset pass {epoch} ──")


def train_epoch_end_line(epoch: int, steps_in_epoch: int) -> str:
    return training_line("epoch", f"── pass {epoch} complete ({steps_in_epoch} steps) ──")


def train_warmup_done_line(step: int) -> str:
    return training_line("warmup", f"LR warmup finished at step {step}")


def train_save_line(phase: str) -> str:
    return training_line("save", phase)


def train_finished_line(steps: int) -> str:
    return training_line("done", f"Training complete after {steps} optimizer steps *")


def framework_banner(title: str) -> str:
    inner_width = min(48, max(12, len(title) + 4))
    bar = "═" * inner_width
    inner = f" {title} "
    if len(inner) < inner_width:
        inner = inner + " " * (inner_width - len(inner))
    elif len(inner) > inner_width:
        inner = inner[: inner_width - 1] + " "
    return f"╔{bar}╗\n║{inner}║\n╚{bar}╝"


def steps_per_epoch(num_examples: int, batch_size: int) -> int:
    return max(1, math.ceil(num_examples / max(1, batch_size)))
