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

export function jobArchitectureSummary(job: TrainingJob): string {
  const c = parseJobConfig(job);
  const arch = c?.architecture as Record<string, number> | undefined;
  if (!arch) return job.model_name;
  return `d=${arch.hidden_size} · L=${arch.num_hidden_layers} · H=${arch.num_attention_heads}`;
}
