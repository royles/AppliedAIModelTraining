import type { TrainingDatasetConfig, TokenizerBuildConfig } from "./api";
import { datasetConfigLooksComplete } from "./datasetValidation";

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

function datasetStepDone(ctx: BlueprintContext): boolean {
  const configured = datasetConfigLooksComplete(ctx.dataset);
  if (ctx.initMode === "from_scratch" && ctx.tokenizer.mode === "train_bpe_from_dataset") {
    if (ctx.dataset.source === "builtin") return false;
    return configured;
  }
  if (ctx.dataset.source === "inline") {
    return (ctx.dataset.texts?.length ?? 0) > 0;
  }
  if (ctx.dataset.source === "builtin") return true;
  return configured;
}

function tokenizerStepDone(ctx: BlueprintContext): boolean {
  if (ctx.initMode !== "from_scratch") return true;
  if (ctx.tokenizer.mode === "pretrained_gpt2") {
    return ctx.vocabSize >= 50257;
  }
  if (ctx.tokenizer.mode !== "train_bpe_from_dataset") return true;
  const tokVocab = ctx.tokenizer.vocab_size ?? ctx.vocabSize;
  return (
    tokVocab >= 256 &&
    tokVocab <= 256_000 &&
    (ctx.tokenizer.min_frequency ?? 2) >= 1 &&
    ctx.dataset.source !== "builtin" &&
    datasetConfigLooksComplete(ctx.dataset)
  );
}

function architectureStepDone(ctx: BlueprintContext): boolean {
  return (
    ctx.hiddenSize > 0 &&
    ctx.layers > 0 &&
    ctx.heads > 0 &&
    ctx.hiddenSize % ctx.heads === 0
  );
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
        "or Parquet on S3. Corpus loading and size checks run when the job starts.",
      dependsOn: ["mode"],
      requirement:
        ctx.initMode === "from_scratch" && ctx.tokenizer.mode === "train_bpe_from_dataset"
          ? "Non-demo source with required fields filled in (HF id, file path, etc.)."
          : "Required fields filled in for the chosen source.",
      volumetrics: `${ctx.dataset.source} · validated at job run`,
      done: datasetStepDone(ctx),
    },
    {
      id: "tokenizer",
      title: "Tokenizer & vocabulary",
      description:
        "The tokenizer maps text to token ids your model will use. " +
        "BPE is trained on your corpus when the job runs.",
      dependsOn: ["dataset"],
      requirement:
        ctx.tokenizer.mode === "train_bpe_from_dataset"
          ? "BPE options set; target vocab in 256–256k."
          : "GPT-2 BPE vocabulary (50,257) or saved artifact.",
      volumetrics: `vocab_size ${(ctx.tokenizer.vocab_size ?? ctx.vocabSize).toLocaleString()} · min_frequency ${ctx.tokenizer.min_frequency ?? 2}`,
      done: tokenizerStepDone(ctx),
    },
    {
      id: "architecture",
      title: "Transformer geometry",
      description:
        "Set hidden size, layer count, and attention heads (hidden size must divide evenly by heads). " +
        "Presets provide starting points; the diagram summarizes capacity and parameter count.",
      dependsOn: ["tokenizer"],
      requirement: "hidden_size divisible by num_attention_heads.",
      volumetrics: `d=${ctx.hiddenSize} · L=${ctx.layers} · H=${ctx.heads} · ~${
        ctx.paramEstimate?.toLocaleString() ?? "?"
      } params`,
      done: architectureStepDone(ctx),
    },
    {
      id: "training",
      title: "Optimization schedule",
      description:
        "max_steps caps how many optimizer updates Ray Train will run. " +
        "Short runs are fine for plumbing checks; meaningful from-scratch models usually need thousands of steps or more.",
      dependsOn: ["architecture"],
      requirement: "max_steps between 10 and 500,000.",
      volumetrics: `max_steps ${ctx.maxSteps.toLocaleString()}`,
      done: ctx.maxSteps >= 10 && ctx.maxSteps <= 500_000,
    },
    {
      id: "compute",
      title: "Ray / CDSW compute",
      description:
        "Workers and GPUs per worker control distributed Ray Train and optional Cloudera AI engines.",
      dependsOn: ["training"],
      requirement: "1–32 workers; GPUs per worker ≥ 1.",
      volumetrics: `${ctx.jobWorkers} worker(s) · ${ctx.jobGpus} GPU/worker`,
      done: ctx.jobWorkers >= 1 && ctx.jobWorkers <= 32 && ctx.jobGpus >= 1,
    },
    {
      id: "register",
      title: "Job metadata & /v1 id",
      description:
        "Name the job and set the OpenAI-compatible model id used when this run registers in the registry.",
      dependsOn: ["compute"],
      requirement: "Job name and OpenAI-compatible model id for registration after success.",
      volumetrics: "model id 1–64 chars: letters, digits, . _ -",
      done: ctx.jobName.trim().length > 0 && OPENAI_ID.test(ctx.openaiModelId.trim()),
    },
  ];

  const doneMap = new Map<string, boolean>();
  return steps.map((step) => {
    doneMap.set(step.id, step.done);
    const waitingOn = step.dependsOn.filter((id) => !doneMap.get(id));
    return { ...step, waitingOn, done: step.done };
  });
}

/** Local field checks only — no dataset download or preview. */
export function wizardSettingsValid(ctx: BlueprintContext): boolean {
  return buildTrainingBlueprint(ctx).every((s) => s.done);
}

export function blueprintReady(
  steps: BlueprintStep[],
  confirmedStepIds?: ReadonlySet<string>,
): boolean {
  if (steps.length === 0) return false;
  return steps.every((s) => s.done && (confirmedStepIds ? confirmedStepIds.has(s.id) : true));
}
