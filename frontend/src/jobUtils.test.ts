import { describe, expect, it } from "vitest";
import type { TrainingJob } from "./api";
import { trainingJobListFieldsEqual } from "./jobUtils";

function job(partial: Partial<TrainingJob> & { id: number }): TrainingJob {
  return {
    id: partial.id,
    experiment_id: 1,
    name: "t",
    status: partial.status ?? "running",
    model_name: "m",
    num_workers: 1,
    gpus_per_worker: 1,
    epochs: 1,
    batch_size: 1,
    learning_rate: 1e-4,
    max_steps: 10,
    config_json: null,
    ray_address: null,
    metrics_json: partial.metrics_json ?? null,
    error_message: partial.error_message ?? null,
    openai_model_id: partial.openai_model_id ?? null,
    created_at: "2026-01-01",
    updated_at: partial.updated_at ?? "2026-01-01",
  };
}

describe("trainingJobListFieldsEqual", () => {
  it("treats matching poll fields as equal", () => {
    const a = job({ id: 1, status: "running" });
    const b = job({ id: 1, status: "running" });
    expect(trainingJobListFieldsEqual(a, b)).toBe(true);
  });

  it("detects status changes", () => {
    const a = job({ id: 1, status: "running" });
    const b = job({ id: 1, status: "succeeded" });
    expect(trainingJobListFieldsEqual(a, b)).toBe(false);
  });
});
