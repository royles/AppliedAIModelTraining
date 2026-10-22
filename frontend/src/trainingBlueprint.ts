import type { TrainingDatasetConfig, TokenizerBuildConfig } from "./api";
import { datasetConfigLooksComplete } from "./datasetValidation";

export type DatasetPreview = {
  examples: number;
  total_characters: number;
  source: string;
  preview_capped?: boolean;
  sample_limit?: number | null;
};

export type BlueprintContext = {
  initMode: "from_scratch" | "pretrained";
  tokenizer: TokenizerBuildConfig;
  vocabSize: number;
  hiddenSize: number;
  layers: number;
  heads: number;
  maxSteps: number;
  jobWorkers: number;
  jobGpus: number;
  dataset: TrainingDatasetConfig;
  selectedProjectId: number | null;
  jobName: string;
  openaiModelId: string;
  paramEstimate: number | null;
  datasetPreview: DatasetPreview | null;
  datasetPreviewLoading?: boolean;
  datasetPreviewError?: string | null;
  minCorpusChars: number | null;
};

export type BlueprintStep = {
  id: string;
  title: string;
  description: string;
  dependsOn: string[];
  requirement: string;
  volumetrics: string;
  done: boolean;
  waitingOn: string[];
};

const OPENAI_ID = /^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$/;

function corpusChars(ctx: BlueprintContext): number {
  if (!ctx.datasetPreview) return 0;
  return ctx.datasetPreview.total_characters;
}

export function buildTrainingBlueprint(ctx: BlueprintContext): BlueprintStep[] {
  const steps: Omit<BlueprintStep, "waitingOn">[] = [
    {
      id: "project",
      title: "Project & jobs",
      description:
        "Projects are collections of models. Each job under a project runs training and can produce a registered model. " +
        "Pick the active project here, then configure the blueprint below and start a new job—or re-run an existing job to train again.",
      dependsOn: [],
      requirement: "Select a project for this job (create one if empty).",
      volumetrics: "One project record in the platform database.",
      done: ctx.selectedProjectId !== null,
    },
    {
      id: "mode",
      title: "Training mode",
      description:
        "Choose whether to train from random weights or fine-tune a Hugging Face checkpoint. " +
        "From scratch needs a corpus-fit tokenizer; fine-tuning reuses preset weights and is usually faster to useful quality.",
      dependsOn: [],
      requirement:
        ctx.initMode === "from_scratch"
          ? "Full from scratch: random weights + corpus-fit tokenizer."
          : "Fine-tune: load a pretrained Hugging Face checkpoint from the preset.",
      volumetrics:
        ctx.initMode === "from_scratch"
          ? "No base checkpoint weights loaded."
          : "Preset supplies base_model (e.g. distilgpt2, gpt2).",
      done: true,
    },
    {
      id: "dataset",
      title: "Training dataset",
      description:
        "Point training at inline text, files under data/datasets, Hugging Face, Iceberg tables, " +
        "or Parquet on S3. From-scratch BPE needs enough real text—built-in demo is for smoke tests only.",
      dependsOn: ["mode"],
      requirement:
        ctx.initMode === "from_scratch" && ctx.tokenizer.mode === "train_bpe_from_dataset"
          ? "Non-demo corpus: files, HF, Iceberg, or S3 Parquet with a successful preview."
          : "Any configured source (builtin OK for smoke tests).",
      volumetrics: ctx.datasetPreview
        ? `${ctx.datasetPreview.examples.toLocaleString()} examples · ${corpusChars(ctx).toLocaleString()} characters (after repeat)`
        : "Load preview to measure examples and characters.",
      done: (() => {
        const configured = datasetConfigLooksComplete(ctx.dataset);
        if (ctx.initMode === "from_scratch" && ctx.tokenizer.mode === "train_bpe_from_dataset") {
          if (ctx.dataset.source === "builtin") return false;
          if (!configured) return false;
          if (ctx.datasetPreviewLoading) return true;
          if (ctx.datasetPreviewError) return true;
          if (!ctx.datasetPreview) return true;
          return ctx.datasetPreview.examples >= 1;
        }
        if (ctx.datasetPreviewError && ctx.dataset.source !== "builtin") {
          return configured;
        }
        if (ctx.dataset.source === "inline") {
          return (ctx.dataset.texts?.length ?? 0) > 0;
        }
        if (ctx.dataset.source === "builtin") return true;
        return configured;
      })(),
    },
    {
      id: "tokenizer",
      title: "Tokenizer & vocabulary",
      description:
        "The tokenizer maps text to token ids your model will use. " +
        "Train byte-pair encoding on your corpus or reuse GPT-2 vocabulary; vocab size must match the architecture step.",
      dependsOn: ["dataset"],
      requirement:
        ctx.tokenizer.mode === "train_bpe_from_dataset"
          ? `BPE trained on corpus; vocab_size matches architecture (${ctx.vocabSize}).`
          : "GPT-2 BPE vocabulary (50,257) or saved artifact.",
      volumetrics: `vocab_size ${ctx.vocabSize.toLocaleString()} · min_frequency ${ctx.tokenizer.min_frequency ?? 2}${
        ctx.minCorpusChars ? ` · need ≥${ctx.minCorpusChars.toLocaleString()} chars` : ""
      }`,
      done: (() => {
        if (ctx.initMode !== "from_scratch") return true;
        if (ctx.tokenizer.mode === "pretrained_gpt2") return ctx.vocabSize >= 50257;
        if (ctx.tokenizer.mode !== "train_bpe_from_dataset") return true;
        const min = ctx.minCorpusChars ?? 10_000;
        const vocabOk = ctx.vocabSize === (ctx.tokenizer.vocab_size ?? ctx.vocabSize);
        if (ctx.dataset.source === "builtin") return false;
        if (!datasetConfigLooksComplete(ctx.dataset)) return false;
        if (ctx.datasetPreviewLoading) return vocabOk;
        if (ctx.datasetPreviewError) return vocabOk;
        if (!ctx.datasetPreview) return vocabOk;
        return vocabOk && corpusChars(ctx) >= min;
      })(),
    },
    {
      id: "architecture",
      title: "Transformer geometry",
      description:
        "Set hidden size, layer count, and attention heads (hidden size must divide evenly by heads). " +
        "Presets provide starting points; the diagram summarizes capacity and parameter count.",
      dependsOn: ["tokenizer"],
      requirement: "hidden_size divisible by num_attention_heads; GPT-2 family for scratch causal LM.",
      volumetrics: `d=${ctx.hiddenSize} · L=${ctx.layers} · H=${ctx.heads} · ~${
        ctx.paramEstimate?.toLocaleString() ?? "?"
      } params`,
      done: (() => {
        const geomOk =
          ctx.hiddenSize > 0 &&
          ctx.layers > 0 &&
          ctx.heads > 0 &&
          ctx.hiddenSize % ctx.heads === 0;
        if (ctx.initMode === "pretrained") {
          return geomOk;
        }
        return geomOk && ctx.paramEstimate !== null && ctx.paramEstimate < 500_000_000;
      })(),
    },
    {
      id: "training",
      title: "Optimization schedule",
      description:
        "max_steps caps how many optimizer updates Ray Train will run. " +
        "Short runs are fine for plumbing checks; meaningful from-scratch models usually need thousands of steps or more.",
      dependsOn: ["architecture"],
      requirement: "max_steps and learning rate set; logging_steps implied by preset.",
      volumetrics: `max_steps ${ctx.maxSteps.toLocaleString()} (typical scratch smoke: 40–500 · serious: 5k–50k+)`,
      done: ctx.maxSteps >= 10 && ctx.maxSteps <= 500_000,
    },
    {
      id: "compute",
      title: "Ray / CDSW compute",
      description:
        "Workers and GPUs per worker control distributed Ray Train and optional Cloudera AI engines. " +
        "Larger models generally need at least one GPU per worker unless you stay on a single worker.",
      dependsOn: ["training"],
      requirement: "Workers and GPUs per worker for Ray Train + optional CDSW engines.",
      volumetrics: `${ctx.jobWorkers} worker(s) · ${ctx.jobGpus} GPU/worker (GPU recommended when params > ~50M)`,
      done:
        ctx.jobWorkers >= 1 &&
        ctx.jobWorkers <= 32 &&
        (ctx.paramEstimate === null ||
          ctx.paramEstimate < 30_000_000 ||
          ctx.jobGpus > 0 ||
          ctx.jobWorkers === 1),
    },
    {
      id: "register",
      title: "Job metadata & /v1 id",
      description:
        "Name the job and set the OpenAI-compatible model id used when this run registers in the registry. " +
        "Re-running the job trains again; a new registry entry may be created depending on lifecycle rules.",
      dependsOn: ["compute"],
      requirement: "Job name and OpenAI-compatible model id for registration after success.",
      volumetrics: "model id 1–64 chars: letters, digits, . _ -",
      done: ctx.jobName.trim().length > 0 && OPENAI_ID.test(ctx.openaiModelId.trim()),
    },
  ];

  const doneMap = new Map<string, boolean>();
  return steps.map((step) => {
    const waitingOn = step.dependsOn.filter((id) => !doneMap.get(id));
    const blocked = waitingOn.length > 0;
    const done = !blocked && step.done;
    doneMap.set(step.id, done);
    return { ...step, waitingOn, done };
  });
}

export function corpusMeetsMinimum(ctx: BlueprintContext): boolean {
  if (ctx.initMode !== "from_scratch") return true;
  if (ctx.tokenizer.mode !== "train_bpe_from_dataset") return true;
  if (ctx.dataset.source === "builtin") return false;
  if (!ctx.datasetPreview || ctx.datasetPreviewLoading) return false;
  const min = ctx.minCorpusChars ?? 10_000;
  return corpusChars(ctx) >= min;
}

export function blueprintReady(
  steps: BlueprintStep[],
  confirmedStepIds?: ReadonlySet<string>,
  ctx?: BlueprintContext,
): boolean {
  if (steps.length === 0) return false;
  const stepsOk = steps.every(
    (s) => s.done && (confirmedStepIds ? confirmedStepIds.has(s.id) : true),
  );
  if (!stepsOk) return false;
  if (ctx && !corpusMeetsMinimum(ctx)) return false;
  return true;
}
