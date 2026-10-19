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
    from app.inference.prompting import PLATFORM_CHAT_TEMPLATE, ensure_platform_chat_template

    if config.get("task_type") == "causal_lm":
        ensure_platform_chat_template(tokenizer)
        if not getattr(tokenizer, "chat_template", None):
            tokenizer.chat_template = PLATFORM_CHAT_TEMPLATE
    tokenizer.save_pretrained(path)
    meta = {
        "init_mode": config.get("init_mode"),
        "task_type": config.get("task_type"),
        "architecture": config.get("architecture"),
        "prompt_format": "platform_user_assistant",
    }
    (path / "platform_config.json").write_text(json.dumps(meta, indent=2))


def _train_loop(config: dict[str, Any]) -> None:
    import ray
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from app.training.model_builder import build_model_and_tokenizer

    task = config["task_type"]
    batch_size = config["per_device_train_batch_size"]
    max_steps = int(config["max_steps"])
    resume_step = int(config.get("resume_global_step") or 0)
    max_len = config["max_seq_length"]
    artifact_dir = config["artifact_dir"]

    tokenizer, model = build_model_and_tokenizer(config)
    tokenizer.padding_side = config.get("padding_side", "right")

    _apply_dropout_overrides(model, config)
    if config.get("gradient_checkpointing") and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    model = _maybe_apply_lora(model, config)

    from app.training.dataset_loader import load_training_samples

    dataset_cfg = config.get("dataset") or {"source": "builtin", "builtin_name": "platform_demo"}
    texts, label_list = load_training_samples(dataset_cfg, task_type=task)
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
        labels = label_list or [i % config["num_labels"] for i in range(len(texts))]
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
    log_every = max(1, int(config.get("logging_steps", 5)))
    warmup_steps = max(1, int(max_steps * config.get("warmup_ratio", 0.03)))
    job_epochs = int(config.get("job_epochs") or 1)

    step = resume_step
    micro = 0
    model.train()
    optimizer.zero_grad()
    if resume_step > 0:
        for _ in range(resume_step):
            scheduler.step()

    job_id = int(config.get("job_id") or 0)
    from app.services.job_log import append_training_log_file
    from app.services.log_flair import (
        steps_per_epoch,
        train_epoch_end_line,
        train_epoch_start_line,
        train_finished_line,
        train_save_line,
        train_step_line,
        train_warmup_done_line,
        train_worker_ready_line,
        training_line,
    )

    def _log_training(message: str, *, all_ranks: bool = False) -> None:
        if job_id <= 0:
            return
        try:
            import ray as _ray

            if not all_ranks and _ray.train.get_context().get_world_rank() != 0:
                return
        except Exception:  # noqa: BLE001
            if not all_ranks:
                return
        append_training_log_file(job_id, message)

    try:
        ctx = ray.train.get_context()
        rank = ctx.get_world_rank()
        world_size = ctx.get_world_size()
    except Exception:  # noqa: BLE001
        rank, world_size = 0, 1

    spe = steps_per_epoch(len(texts), batch_size)
    ds = config.get("dataset") or {}
    _log_training(
        train_worker_ready_line(rank, world_size, str(device), len(texts), max_steps),
        all_ranks=True,
    )
    if rank == 0:
        _log_training(
            training_line(
                "info",
                f"task={task} batch={batch_size} seq={max_len} "
                f"dataset={ds.get('source', 'builtin')} steps/epoch~{spe} "
                f"target_epochs={job_epochs} warmup_steps={warmup_steps}",
            )
        )

    def _should_log_step(s: int) -> bool:
        if s == 1 or s >= max_steps:
            return True
        if max_steps <= 200:
            return True
        return s % log_every == 0

    dataset_pass = int(config.get("resume_epoch") or 0)
    warmup_logged = resume_step >= warmup_steps

    while step < max_steps:
        dataset_pass += 1
        if rank == 0:
            _log_training(train_epoch_start_line(dataset_pass))
        steps_this_pass = 0
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
                steps_this_pass += 1
                loss_val = float(loss.item() * accum)
                lr_val = scheduler.get_last_lr()[0]
                if rank == 0 and _should_log_step(step):
                    ray.train.report(
                        {
                            "train_loss": loss_val,
                            "step": step,
                            "lr": lr_val,
                            "epoch": dataset_pass,
                        }
                    )
                    _log_training(
                        train_step_line(
                            step,
                            max_steps,
                            loss_val,
                            lr_val,
                            dataset_pass,
                            spe,
                        )
                    )
                if rank == 0 and not warmup_logged and step >= warmup_steps:
                    warmup_logged = True
                    _log_training(train_warmup_done_line(step))
                if step >= max_steps:
                    break
        if rank == 0 and steps_this_pass > 0:
            _log_training(train_epoch_end_line(dataset_pass, steps_this_pass))
            if task == "causal_lm":
                from app.services.job_progress import write_progress

                ckpt_dir = Path(artifact_dir) / "checkpoints" / f"epoch_{dataset_pass}"
                _save_artifacts(model, tokenizer, str(ckpt_dir), config)
                write_progress(
                    artifact_dir,
                    {
                        "global_step": step,
                        "epoch": dataset_pass,
                        "max_steps": max_steps,
                    },
                )
                _log_training(train_save_line(f"Checkpoint saved (epoch {dataset_pass}, step {step})."))
        if step >= max_steps:
            break

    if rank == 0:
        _log_training(train_finished_line(step))
    if rank == 0 and task == "causal_lm":
        _log_training(train_save_line("Writing model + tokenizer to artifact_dir…"))
        _save_artifacts(model, tokenizer, artifact_dir, config)
        _log_training(train_save_line("Artifacts saved."))


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

    job_id = int(train_config.get("job_id") or 0)

    from app.services.log_flair import framework_line
    from app.services.ray_cluster_watch import log_ray_nodes_snapshot, wait_for_ray_nodes

    def _framework(msg: str) -> None:
        if job_id <= 0:
            return
        try:
            from app.database import SessionLocal
            from app.services.job_log import append_job_log

            db = SessionLocal()
            try:
                append_job_log(db, job_id, "framework", msg)
            finally:
                db.close()
        except Exception:  # noqa: BLE001
            pass

    if ray_address:
        from app.services.cdsw_workers import ray_driver_init_address

        ray_address = ray_driver_init_address(ray_address) or ray_address
        os.environ["RAY_ADDRESS"] = ray_address
        worker_head = train_config.get("ray_head_address")
        if worker_head and worker_head != ray_address:
            _framework(
                framework_line(
                    "ray",
                    f"Driver connecting via {ray_address} (CDSW workers join {worker_head})",
                )
            )
        else:
            _framework(framework_line("ray", f"Driver connecting to Ray at {ray_address}"))
        from app.services.cdsw_workers import wait_for_gcs_port

        host, _, port_s = ray_address.partition(":")
        port = int(port_s or "6379")
        if not wait_for_gcs_port(host, port, timeout_sec=180.0):
            raise RuntimeError(f"Ray GCS not reachable at {ray_address} before ray.init")
        if not ray.is_initialized():
            ray.init(address=ray_address, ignore_reinit_error=True)
    else:
        _framework(framework_line("ray", "Starting local Ray runtime"))
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

    log_ray_nodes_snapshot(_framework, "Driver view")
    expected_cdsw = int(train_config.get("expected_cdsw_worker_nodes") or 0)
    if expected_cdsw > 0:
        wait_for_ray_nodes(
            _framework,
            min_alive=1 + expected_cdsw,
            timeout_sec=600.0,
        )
    else:
        try:
            resources = ray.cluster_resources()
            _framework(
                framework_line(
                    "ray",
                    f"Cluster GPUs={resources.get('GPU', 0)} CPUs={resources.get('CPU', 0)}",
                )
            )
        except Exception as exc:  # noqa: BLE001
            _framework(framework_line("warn", f"Cluster stats unavailable: {exc}"))

    from app.services.ray_train_runtime import (
        apply_ray_train_runtime_env,
        cluster_resource_summary,
        verify_train_resources,
    )

    apply_ray_train_runtime_env()
    resources = cluster_resource_summary()
    if resources:
        _framework(
            framework_line(
                "ray",
                "Cluster resources before Train: "
                + ", ".join(f"{k}={v:g}" for k, v in sorted(resources.items())),
            )
        )
    try:
        verify_train_resources(
            num_workers=num_workers,
            use_gpu=use_gpu,
            cpus_per_worker=1,
            gpus_per_worker=1 if use_gpu else 0,
        )
    except RuntimeError as exc:
        _framework(framework_line("fail", str(exc)))
        raise

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
    _framework(
        framework_line(
            "train",
            f"Scheduling {num_workers} Ray Train worker(s) on master gpu={use_gpu}",
        )
    )
    result = trainer.fit()
    _framework(framework_line("done", "Ray Train run complete; collecting metrics"))
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
        from app.services.cdsw_workers import ray_driver_init_address

        return ray_driver_init_address(explicit) or explicit
    env = os.environ.get("RAY_ADDRESS")
    if env:
        return env
    head = os.environ.get("RAY_HEAD_ADDRESS")
    if head:
        from app.services.cdsw_workers import ray_client_address

        return ray_client_address(head)
    return None
