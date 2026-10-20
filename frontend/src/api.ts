/** Training project (groups model tasks). API table name remains `experiments`. */
export type Project = {
  id: number;
  name: string;
  description: string | null;
  created_at: string;
};

/** @deprecated Use Project */
export type Experiment = Project;

export type TransformerArchitecture = {
  family: "gpt2" | "bert";
  vocab_size: number;
  hidden_size: number;
  num_hidden_layers: number;
  num_attention_heads: number;
  intermediate_size?: number | null;
  max_position_embeddings: number;
  tie_word_embeddings: boolean;
};

export type TokenizerBuildConfig = {
  mode: "train_bpe_from_dataset" | "pretrained_gpt2" | "saved_artifact";
  vocab_size?: number;
  min_frequency?: number;
  min_corpus_chars?: number | null;
};

export type TrainingDatasetConfig = {
  source: "builtin" | "inline" | "text_file" | "jsonl_file" | "huggingface";
  builtin_name?: string;
  texts?: string[];
  labels?: number[] | null;
  path?: string | null;
  text_field?: string;
  label_field?: string | null;
  hf_dataset?: string | null;
  hf_split?: string;
  hf_text_column?: string;
  hf_label_column?: string | null;
  max_samples?: number | null;
  repeat?: number;
};

export type TransformerConfig = {
  init_mode: "from_scratch" | "pretrained";
  task_type: "causal_lm" | "sequence_classification";
  architecture: TransformerArchitecture;
  base_model?: string | null;
  max_seq_length: number;
  learning_rate: number;
  per_device_train_batch_size: number;
  max_steps: number;
  openai_model_id?: string | null;
  dataset?: TrainingDatasetConfig;
  tokenizer?: TokenizerBuildConfig;
  use_lora?: boolean;
};

export type TransformerPreset = {
  name: string;
  title: string;
  description: string;
  config: TransformerConfig;
};

export type TrainingJob = {
  id: number;
  experiment_id: number;
  name: string;
  status: string;
  model_name: string;
  num_workers: number;
  gpus_per_worker: number;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  max_steps: number;
  config_json: string | null;
  ray_address: string | null;
  metrics_json: string | null;
  error_message: string | null;
  openai_model_id: string | null;
  created_at: string;
  updated_at: string;
};

export type Health = {
  status: string;
  ray_runtime: string;
  cdsw_available: boolean;
  details: Record<string, unknown>;
};

export type RaySettings = {
  ray_local_mode: boolean;
  ray_head_address: string | null;
  ray_train_num_workers: number;
  non_distributed_low_memory_mode: boolean;
};

export type CdswWorkerSettings = {
  worker_cpu: number;
  worker_memory_gb: number;
  gpus_per_worker: number;
  default_job_num_workers: number;
  accelerator_label_id: string | null;
  resource_profile_id: string | null;
  resource_group_id: string | null;
};

export type PlatformRuntimeInfo = {
  ray_runtime: string;
  cdsw_available: boolean;
  detected_ray_head: string | null;
};

export type PlatformSettings = {
  ray: RaySettings;
  cdsw_workers: CdswWorkerSettings;
  runtime: PlatformRuntimeInfo;
};

export type PlatformSettingsUpdate = {
  ray?: RaySettings;
  cdsw_workers?: CdswWorkerSettings;
};

export type LogLine = {
  ts: string;
  message: string;
};

export type JobLogs = {
  job_id: number;
  status: string;
  ray_address: string | null;
  framework: LogLine[];
  training: LogLine[];
  ray_status: Record<string, unknown>;
  metrics: Record<string, unknown> | null;
};

export type RegistryLifecycleInfo = {
  state: string;
  guidance: string;
  transitions_to: string[];
};

export type RegistryModel = {
  id: number;
  job_id: number;
  job_name: string | null;
  openai_model_id: string;
  display_name: string | null;
  description: string | null;
  lifecycle_state: string;
  task_type: string;
  base_model: string;
  artifact_path: string;
  created_at_unix: number;
  updated_at: string;
  parent_model_id: number | null;
  metadata: Record<string, unknown>;
  benchmark: Record<string, unknown> | null;
  model_card_markdown: string | null;
};

/** Resolve API paths against the current page URL (works behind CAI path prefixes). */
export function resolveApiUrl(path: string): string {
  const relative = path.startsWith("/") ? path.slice(1) : path;
  return new URL(relative, window.location.href).toString();
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(resolveApiUrl(path), {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  const text = await res.text();
  if (!res.ok) {
    let message = text || res.statusText;
    try {
      const body = JSON.parse(text) as { detail?: string | { msg?: string }[] };
      if (typeof body.detail === "string") {
        message = body.detail;
      } else if (Array.isArray(body.detail)) {
        message = body.detail.map((d) => d.msg ?? JSON.stringify(d)).join("; ");
      }
    } catch {
      /* keep raw text */
    }
    throw new Error(message);
  }
  try {
    return JSON.parse(text) as T;
  } catch {
    throw new Error(
      `Expected JSON from ${path} but received HTML or non-JSON. ` +
        "Check that start.py is running app.ui_gateway on the Application port and open /api/health in the browser.",
    );
  }
}

export const api = {
  health: () => request<Health>("/api/health"),
  getPlatformSettings: () => request<PlatformSettings>("/api/platform/settings"),
  updatePlatformSettings: (body: PlatformSettingsUpdate) =>
    request<PlatformSettings>("/api/platform/settings", {
      method: "PUT",
      body: JSON.stringify(body),
    }),
  uiBuildInfo: () =>
    request<{
      status: string;
      platform_version?: string;
      js?: string;
      css?: string;
      stamp?: string;
      features?: { model_registry?: boolean; registry_chat?: boolean };
    }>("/api/ui/build-info"),
  listPresets: () => request<TransformerPreset[]>("/api/transformer/presets"),
  listDatasetFiles: () =>
    request<{ path: string; kind: string; size_bytes: string }[]>("/api/training/datasets/files"),
  listBuiltinDatasets: () =>
    request<Record<string, { title: string; description: string }>>("/api/training/datasets/builtin"),
  fromScratchGuidance: (body: Partial<TransformerConfig> & { init_mode: string; task_type: string }) =>
    request<Record<string, unknown>>("/api/training/from-scratch/guidance", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  datasetPreview: (dataset: TrainingDatasetConfig) =>
    request<{ examples: number; total_characters: number; source: string }>(
      "/api/training/dataset/preview",
      { method: "POST", body: JSON.stringify(dataset) },
    ),
  estimateArchitecture: (architecture: TransformerArchitecture) =>
    request<{ estimated_parameters: number }>("/api/transformer/architecture/estimate", {
      method: "POST",
      body: JSON.stringify(architecture),
    }),
  listProjects: () => request<Project[]>("/api/projects"),
  createProject: (body: { name: string; description?: string }) =>
    request<Project>("/api/projects", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  updateProject: (projectId: number, body: { name?: string; description?: string | null }) =>
    request<Project>(`/api/projects/${projectId}`, {
      method: "PATCH",
      body: JSON.stringify(body),
    }),
  deleteProject: (projectId: number) =>
    request<{ status: string; project_id: string }>(`/api/projects/${projectId}`, {
      method: "DELETE",
    }),
  /** @deprecated Use listProjects */
  listExperiments: () => request<Project[]>("/api/experiments"),
  /** @deprecated Use createProject */
  createExperiment: (body: { name: string; description?: string }) =>
    request<Project>("/api/experiments", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  listJobs: () => request<TrainingJob[]>("/api/jobs"),
  getJob: (jobId: number) => request<TrainingJob>(`/api/jobs/${jobId}`),
  getJobLogs: (jobId: number) => request<JobLogs>(`/api/jobs/${jobId}/logs`),
  deleteJob: (jobId: number) =>
    request<{ status: string; job_id: string }>(`/api/jobs/${jobId}`, {
      method: "DELETE",
    }),
  updateJob: (jobId: number, body: { name?: string; experiment_id?: number; openai_model_id?: string }) =>
    request<TrainingJob>(`/api/jobs/${jobId}`, {
      method: "PATCH",
      body: JSON.stringify(body),
    }),
  restartJob: (jobId: number, body?: { resume?: boolean }) =>
    request<TrainingJob>(`/api/jobs/${jobId}/restart`, {
      method: "POST",
      body: JSON.stringify(body ?? { resume: true }),
    }),
  createJob: (body: {
    experiment_id: number;
    name: string;
    preset?: string;
    transformer?: Partial<TransformerConfig>;
    openai_model_id?: string;
    num_workers?: number;
    gpus_per_worker?: number;
    max_steps?: number;
  }) =>
    request<TrainingJob>("/api/jobs", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  scaleJob: (jobId: number, target_workers: number) =>
    request<{ id: number; target_workers: number; status: string }>(
      `/api/jobs/${jobId}/scale`,
      { method: "POST", body: JSON.stringify({ target_workers }) },
    ),
  listRegistryModels: () => request<RegistryModel[]>("/api/registry/models"),
  getRegistryModel: (id: number) => request<RegistryModel>(`/api/registry/models/${id}`),
  registryLifecycle: () => request<RegistryLifecycleInfo[]>("/api/registry/lifecycle"),
  transitionRegistryModel: (id: number, lifecycle_state: string) =>
    request<RegistryModel>(`/api/registry/models/${id}/transition`, {
      method: "POST",
      body: JSON.stringify({ lifecycle_state }),
    }),
};
