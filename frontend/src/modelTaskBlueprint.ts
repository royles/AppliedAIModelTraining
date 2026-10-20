import type { TrainingDatasetConfig, TokenizerBuildConfig, TransformerConfig } from "./api";

export const MODEL_TASK_BLUEPRINT_KIND = "model_task_blueprint";
export const MODEL_TASK_BLUEPRINT_SCHEMA_VERSION = 1;

export type ModelTaskBlueprintFile = {
  schema_version: number;
  kind: typeof MODEL_TASK_BLUEPRINT_KIND;
  name?: string;
  updated_at?: string;
  preset: string;
  project_id: number | null;
  job_name: string;
  openai_model_id: string;
  num_workers: number;
  gpus_per_worker: number;
  max_steps: number;
  init_mode: "from_scratch" | "pretrained";
  task_type: "causal_lm" | "sequence_classification";
  arch_family: "gpt2" | "bert";
  vocab_size: number;
  hidden_size: number;
  layers: number;
  heads: number;
  max_position: number;
  intermediate_size: number;
  tie_word_embeddings: boolean;
  tokenizer: TokenizerBuildConfig;
  dataset: TrainingDatasetConfig;
  transformer?: Partial<TransformerConfig>;
};

export type ModelTaskWizardState = Omit<
  ModelTaskBlueprintFile,
  "schema_version" | "kind" | "updated_at" | "name" | "transformer"
>;

export function buildModelTaskBlueprint(state: ModelTaskWizardState, name?: string): ModelTaskBlueprintFile {
  return {
    schema_version: MODEL_TASK_BLUEPRINT_SCHEMA_VERSION,
    kind: MODEL_TASK_BLUEPRINT_KIND,
    name,
    updated_at: new Date().toISOString(),
    ...state,
  };
}

export function parseModelTaskBlueprintFile(raw: unknown): ModelTaskBlueprintFile {
  if (!raw || typeof raw !== "object") {
    throw new Error("Blueprint file must be a JSON object");
  }
  const o = raw as Record<string, unknown>;
  if (o.kind !== MODEL_TASK_BLUEPRINT_KIND) {
    throw new Error(`Expected kind "${MODEL_TASK_BLUEPRINT_KIND}"`);
  }
  const version = Number(o.schema_version ?? 0);
  if (version !== MODEL_TASK_BLUEPRINT_SCHEMA_VERSION) {
    throw new Error(`Unsupported schema_version ${version}`);
  }
  return o as ModelTaskBlueprintFile;
}

export function wizardStateFromBlueprint(b: ModelTaskBlueprintFile): ModelTaskWizardState {
  return {
    preset: b.preset,
    project_id: b.project_id,
    job_name: b.job_name,
    openai_model_id: b.openai_model_id,
    num_workers: b.num_workers,
    gpus_per_worker: b.gpus_per_worker,
    max_steps: b.max_steps,
    init_mode: b.init_mode,
    task_type: b.task_type,
    arch_family: b.arch_family,
    vocab_size: b.vocab_size,
    hidden_size: b.hidden_size,
    layers: b.layers,
    heads: b.heads,
    max_position: b.max_position,
    intermediate_size: b.intermediate_size,
    tie_word_embeddings: b.tie_word_embeddings,
    tokenizer: b.tokenizer,
    dataset: b.dataset,
  };
}

export function downloadBlueprintJson(blueprint: ModelTaskBlueprintFile, filename?: string) {
  const blob = new Blob([JSON.stringify(blueprint, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename ?? `${blueprint.name?.replace(/\s+/g, "-") || "model-task-blueprint"}.json`;
  a.click();
  URL.revokeObjectURL(url);
}
