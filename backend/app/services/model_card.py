"""Generate Hugging Face–style model cards for the registry."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from app.models import TrainingJob, TrainedModel
from app.services.model_lifecycle import state_guidance


def _parse_json(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def build_training_metadata(
    *,
    job: TrainingJob,
    config_json: str | None,
    metrics: dict[str, Any] | None,
    artifact_path: str,
    init_mode: str | None,
    dataset_summary: str | None = None,
) -> dict[str, Any]:
    cfg = _parse_json(config_json)
    return {
        "job_id": job.id,
        "job_name": job.name,
        "init_mode": init_mode or cfg.get("init_mode"),
        "task_type": cfg.get("task_type"),
        "architecture": cfg.get("architecture"),
        "dataset": cfg.get("dataset"),
        "dataset_summary": dataset_summary,
        "hyperparameters": {
            "epochs": job.epochs,
            "max_steps": job.max_steps,
            "batch_size": job.batch_size,
            "learning_rate": job.learning_rate,
            "num_workers": job.num_workers,
            "gpus_per_worker": job.gpus_per_worker,
        },
        "training_metrics": metrics or _parse_json(job.metrics_json),
        "artifact_path": artifact_path,
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }


def generate_model_card(
    record: TrainedModel,
    *,
    job: TrainingJob | None = None,
    extra_notes: str | None = None,
) -> str:
    meta = _parse_json(record.metadata_json)
    lineage = meta.get("lineage_snapshot") or {}
    arch = _parse_json(record.architecture_json)
    bench = _parse_json(record.benchmark_json)
    job_name = meta.get("job_name") or (job.name if job else f"job #{record.job_id}")
    init_mode = meta.get("init_mode") or "unknown"
    hp = meta.get("hyperparameters") or {}
    metrics = meta.get("training_metrics") or {}
    dataset = meta.get("dataset") or meta.get("dataset_summary") or "See training job config"

    lines = [
        f"# Model card: `{record.openai_model_id}`",
        "",
        f"**Display name:** {record.display_name or job_name}  ",
        f"**Lifecycle:** `{record.lifecycle_state}` — {state_guidance(record.lifecycle_state)}",
        "",
        "## Model description",
        "",
        record.description
        or (
            f"Causal language model produced by platform training job **{job_name}** "
            f"(init_mode=`{init_mode}`)."
        ),
        "",
        "## Intended use",
        "",
        "- **Primary:** Experimental chat / completion via OpenAI-compatible `/v1/chat/completions`.",
        "- **Not for:** Production decisions without independent safety, bias, and quality review.",
        "- **Serving:** Promote to `production` in the Model Registry when approved.",
        "",
        "## Training data",
        "",
        f"- **Source:** {dataset}",
        f"- **Job id:** #{record.job_id}",
        "",
        "## Lineage (immutable snapshot)",
        "",
        f"- **Project:** {lineage.get('project_name') or '—'} (id={lineage.get('project_id', '—')})",
        f"- **Task:** {lineage.get('task_name') or '—'} "
        f"(id={lineage.get('task_id', '—')}, version={lineage.get('task_version', '—')})",
        f"- **Task version id:** {lineage.get('task_version_id', '—')}",
        f"- **Config hash:** `{lineage.get('content_hash') or '—'}`",
        "",
        "## Ray configuration (project snapshot)",
        "",
    ]
    ray_cfg = lineage.get("ray_config") if isinstance(lineage.get("ray_config"), dict) else {}
    ray = ray_cfg.get("ray") if isinstance(ray_cfg.get("ray"), dict) else {}
    cdsw = ray_cfg.get("cdsw_workers") if isinstance(ray_cfg.get("cdsw_workers"), dict) else {}
    lines.extend(
        [
            f"- **Project Ray id:** {ray_cfg.get('project_id', '—')}",
            f"- **Ray local mode:** {ray.get('ray_local_mode', '—')}",
            f"- **Ray head override:** `{ray.get('ray_head_address') or '—'}`",
            f"- **Ray Train worker cap:** {ray.get('ray_train_num_workers', '—')}",
            f"- **Low-memory session mode:** {ray.get('non_distributed_low_memory_mode', '—')}",
            f"- **Worker CPU / memory (project default):** {cdsw.get('worker_cpu', '—')} CPU, "
            f"{cdsw.get('worker_memory_gb', '—')} GiB",
            f"- **GPUs per worker (project default):** {cdsw.get('gpus_per_worker', '—')}",
            "",
            "## Job run (execution snapshot)",
            "",
        ]
    )
    job_run = lineage.get("job_run") if isinstance(lineage.get("job_run"), dict) else {}
    wg = job_run.get("worker_group") if isinstance(job_run.get("worker_group"), dict) else {}
    launch = (
        job_run.get("effective_worker_launch_spec")
        if isinstance(job_run.get("effective_worker_launch_spec"), dict)
        else {}
    )
    lines.extend(
        [
            f"- **Final status:** `{job_run.get('status', '—')}`",
            f"- **Ray address:** `{job_run.get('ray_address') or '—'}`",
            f"- **Workers requested:** {job_run.get('num_workers', '—')} "
            f"(GPU/worker: {job_run.get('gpus_per_worker', '—')})",
            f"- **Effective CDSW launch spec:** {launch.get('cpu', '—')} CPU, "
            f"{launch.get('memory_gb', '—')} GiB, {launch.get('gpus_per_worker', '—')} GPU/engine",
            f"- **Worker group head:** `{wg.get('head_address') or '—'}`",
            f"- **CDSW worker ids:** {wg.get('cdsw_worker_ids') or '—'}",
            f"- **Completed at:** {job_run.get('completed_at') or '—'}",
            "",
            "## Training procedure",
        ]
    )
    lines.extend(
        [
        "",
        f"- **Base / builder:** `{record.base_model}`",
        f"- **Task:** `{record.task_type}`",
        f"- **Steps / epochs:** max_steps={hp.get('max_steps', '—')}, epochs={hp.get('epochs', '—')}",
        f"- **Batch / LR:** batch_size={hp.get('batch_size', '—')}, lr={hp.get('learning_rate', '—')}",
        f"- **Workers:** {hp.get('num_workers', '—')} (GPU/worker: {hp.get('gpus_per_worker', '—')})",
        "",
        "## Architecture (summary)",
        "",
        "```json",
        json.dumps(arch, indent=2) if arch else "{}",
        "```",
        "",
        "## Training metrics",
        "",
        "```json",
        json.dumps(metrics, indent=2) if metrics else "{}",
        "```",
        "",
        ]
    )
    if bench:
        lines.extend(
            [
                "## Benchmarks",
                "",
                "```json",
                json.dumps(bench, indent=2),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Limitations",
            "",
            "- Small or synthetic corpora produce repetitive / low-quality text.",
            "- From-scratch models need substantial data and compute for useful chat.",
            "- Review tokenizer vocabulary and prompt format (`User:` / `Assistant:`) before eval.",
            "",
            "## How to use (API)",
            "",
            "```bash",
            f'curl -s "$BASE/v1/chat/completions" -H "Content-Type: application/json" -d \'{{',
            f'  "model": "{record.openai_model_id}",',
            '  "messages": [{"role": "user", "content": "Hello"}],',
            '  "temperature": 0.5,',
            '  "max_tokens": 128',
            "}'",
            "```",
            "",
            "## Lifecycle guidance",
            "",
            "| State | Meaning |",
            "| --- | --- |",
            "| `trained` | Artifact registered after successful job |",
            "| `testing` | Active evaluation / smoke tests |",
            "| `benchmarked` | Quantitative benchmark recorded |",
            "| `production` | Approved for default serving |",
            "| `retired` | Archived; inference disabled |",
            "",
        ]
    )
    if extra_notes:
        lines.extend(["## Notes", "", extra_notes, ""])
    lines.append(f"_Card generated by Applied AI Model Training platform._")
    return "\n".join(lines)
