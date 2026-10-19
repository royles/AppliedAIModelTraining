import type { TrainingJob } from "./api";

export function parseJobConfig(job: TrainingJob): Record<string, unknown> | null {
  if (!job.config_json) return null;
  try {
    return JSON.parse(job.config_json) as Record<string, unknown>;
  } catch {
    return null;
  }
}

export function jobInitMode(job: TrainingJob): string {
  const c = parseJobConfig(job);
  return (c?.init_mode as string) ?? "—";
}

export function jobDatasetSummary(job: TrainingJob): string {
  const c = parseJobConfig(job);
  const ds = c?.dataset as Record<string, unknown> | undefined;
  if (!ds || ds.source === "builtin") {
    return `builtin:${(ds?.builtin_name as string) ?? "platform_demo"}`;
  }
  if (ds.source === "inline") {
    const texts = ds.texts as string[] | undefined;
    return `inline (${texts?.length ?? 0} examples)`;
  }
  if (ds.source === "huggingface") {
    return `hf:${ds.hf_dataset ?? "?"}`;
  }
  return `${ds.source}:${ds.path ?? "?"}`;
}

export function jobArchitectureSummary(job: TrainingJob): string {
  const c = parseJobConfig(job);
  const arch = c?.architecture as Record<string, number> | undefined;
  if (!arch) return job.model_name;
  return `d=${arch.hidden_size} · L=${arch.num_hidden_layers} · H=${arch.num_attention_heads}`;
}

/** Fields polled from task detail that affect the task list card. */
export function trainingJobListFieldsEqual(a: TrainingJob, b: TrainingJob): boolean {
  return (
    a.status === b.status &&
    a.error_message === b.error_message &&
    a.metrics_json === b.metrics_json &&
    a.openai_model_id === b.openai_model_id &&
    a.updated_at === b.updated_at
  );
}
