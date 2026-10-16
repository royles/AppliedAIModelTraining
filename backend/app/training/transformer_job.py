"""Ray Train jobs for transformer fine-tuning with configurable best-practice hyperparameters."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

def _build_scheduler(optimizer, config: dict[str, Any]):
    import torch
    from torch.optim.lr_scheduler import LambdaLR

    max_steps = config["max_steps"]
    warmup = max(1, int(max_steps * config["warmup_ratio"]))
    sched_type = config.get("lr_scheduler_type", "cosine")

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return float(step) / float(warmup)
        progress = (step - warmup) / max(1, max_steps - warmup)
        progress = min(1.0, max(0.0, progress))
        if sched_type == "constant":
            return 1.0
        if sched_type == "linear":
            return 1.0 - progress
        # cosine
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def _maybe_apply_lora(model, config: dict[str, Any]):
    if not config.get("use_lora"):
        return model
    from peft import LoraConfig, get_peft_model

    task = config["task_type"]
    targets = config.get("lora_target_modules")
    if not targets:
        targets = (
            ["q_proj", "v_proj", "k_proj", "o_proj"]
            if task == "causal_lm"
            else ["q_lin", "v_lin"]
        )
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=targets,
        bias="none",
        task_type="CAUSAL_LM" if task == "causal_lm" else "SEQ_CLS",
    )
    return get_peft_model(model, lora_config)


def _apply_dropout_overrides(model, config: dict[str, Any]) -> None:
    if config.get("attention_dropout") is not None and hasattr(model.config, "attention_dropout"):
        model.config.attention_dropout = config["attention_dropout"]
    if config.get("hidden_dropout") is not None and hasattr(model.config, "hidden_dropout_prob"):
        model.config.hidden_dropout_prob = config["hidden_dropout"]


def _save_artifacts(model, tokenizer, artifact_dir: str, config: dict[str, Any]) -> None:
    import json

    path = Path(artifact_dir)
    path.mkdir(parents=True, exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model
    if hasattr(model_to_save, "save_pretrained"):
        model_to_save.save_pretrained(path)
    else:
        raise RuntimeError("Model does not support save_pretrained")
    tokenizer.save_pretrained(path)
    meta = {
        "init_mode": config.get("init_mode"),
        "task_type": config.get("task_type"),
        "architecture": config.get("architecture"),
    }
    (path / "platform_config.json").write_text(json.dumps(meta, indent=2))


def _sample_texts(task_type: str) -> list[str]:
    if task_type == "causal_lm":
        return [
            "User: How does Ray scale training?\nAssistant: Ray shards workers across GPUs.",
            "User: What is LoRA?\nAssistant: Low-rank adapters that fine-tune efficiently.",
            "User: Explain CDSW workers.\nAssistant: They launch elastic engines on Cloudera AI.",
        ] * 40
    return [
        "ray scales transformer training across GPUs",
        "cdsw workers attach elastic compute to the cluster",
        "local development uses CPU ray workers",
        "cloudera machine learning hosts the control plane",
    ] * 32


def _train_loop(config: dict[str, Any]) -> None:
    import ray
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from app.training.model_builder import build_model_and_tokenizer

    task = config["task_type"]
    batch_size = config["per_device_train_batch_size"]
    max_steps = config["max_steps"]
    max_len = config["max_seq_length"]
    artifact_dir = config["artifact_dir"]

    tokenizer, model = build_model_and_tokenizer(config)
    tokenizer.padding_side = config.get("padding_side", "right")

    _apply_dropout_overrides(model, config)
    if config.get("gradient_checkpointing") and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    model = _maybe_apply_lora(model, config)

    texts = _sample_texts(task)
    if task == "causal_lm":
        enc = tokenizer(
            texts,
            truncation=config.get("truncation", True),
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        dataset = TensorDataset(enc["input_ids"], enc["attention_mask"])
    else:
        labels = [i % config["num_labels"] for i in range(len(texts))]
        enc = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        dataset = TensorDataset(
            enc["input_ids"], enc["attention_mask"], torch.tensor(labels)
        )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loader = ray.train.torch.prepare_data_loader(loader)
    model = ray.train.torch.prepare_model(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    use_fp16 = config.get("fp16") and device.type == "cuda"
    use_bf16 = config.get("bf16") and device.type == "cuda"
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16 if use_fp16 else None

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        betas=(config["adam_beta1"], config["adam_beta2"]),
        eps=config["adam_epsilon"],
        weight_decay=config["weight_decay"],
    )
    scheduler = _build_scheduler(optimizer, config)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=config.get("label_smoothing", 0.0))

    accum = config.get("gradient_accumulation_steps", 1)
    max_grad_norm = config["max_grad_norm"]
    log_every = config.get("logging_steps", 5)

    step = 0
    micro = 0
    model.train()
    optimizer.zero_grad()

    while step < max_steps:
        for batch in loader:
            if task == "causal_lm":
                input_ids, attention_mask = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = input_ids.clone()
                labels[attention_mask == 0] = -100

                context = (
                    torch.autocast(device_type=device.type, dtype=autocast_dtype)
                    if autocast_dtype
                    else torch.enable_grad()
                )
                with context:
                    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = out.loss / accum
            else:
                input_ids, attention_mask, y = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                y = y.to(device)
                context = (
                    torch.autocast(device_type=device.type, dtype=autocast_dtype)
                    if autocast_dtype
                    else torch.enable_grad()
                )
                with context:
                    out = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = loss_fn(out.logits, y) / accum

            loss.backward()
            micro += 1
            if micro % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
                if step % log_every == 0:
                    ray.train.report(
                        {
                            "train_loss": float(loss.item() * accum),
                            "step": step,
                            "lr": scheduler.get_last_lr()[0],
                        }
                    )
                if step >= max_steps:
                    break
        if step >= max_steps:
            break

    ctx = ray.train.get_context()
    if ctx.get_world_rank() == 0 and task == "causal_lm":
        _save_artifacts(model, tokenizer, artifact_dir, config)


def run_transformer_training(
    *,
    ray_address: str | None,
    num_workers: int,
    use_gpu: bool,
    train_config: dict[str, Any],
) -> dict[str, Any]:
    import ray
    from ray.train import ScalingConfig
    from ray.train.torch import TorchTrainer

    if ray_address:
        if not ray.is_initialized():
            ray.init(address=ray_address, ignore_reinit_error=True)
    else:
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

    scaling = ScalingConfig(
        num_workers=num_workers,
        use_gpu=use_gpu,
        resources_per_worker={"CPU": 1, "GPU": 1 if use_gpu else 0},
    )
    trainer = TorchTrainer(
        train_loop_per_worker=_train_loop,
        train_loop_config=train_config,
        scaling_config=scaling,
    )
    result = trainer.fit()
    artifact = train_config.get("artifact_dir")
    return {
        "metrics": result.metrics,
        "checkpoint": str(result.checkpoint),
        "artifact_dir": artifact,
        "task_type": train_config.get("task_type"),
        "base_model": train_config.get("base_model"),
    }


def resolve_ray_address(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    env = os.environ.get("RAY_ADDRESS")
    if env:
        return env
    head = os.environ.get("RAY_HEAD_ADDRESS")
    if head:
        from app.services.cdsw_workers import ray_client_address

        return ray_client_address(head)
    return None
