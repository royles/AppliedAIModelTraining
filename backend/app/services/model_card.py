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
        "## Training procedure",
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
