import type { TrainingDatasetConfig, TokenizerBuildConfig } from "./api";
import { validateDatasetConfig } from "./datasetValidation";

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
  if (!validateDatasetConfig(ctx.dataset).valid) {
    return false;
  }
  if (ctx.initMode === "from_scratch" && ctx.tokenizer.mode === "train_bpe_from_dataset") {
    return ctx.dataset.source !== "builtin";
  }
  return true;
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
    validateDatasetConfig(ctx.dataset).valid
  );
}

function geometryStepDone(ctx: BlueprintContext): boolean {
  return (
    ctx.hiddenSize > 0 &&
    ctx.layers > 0 &&
    ctx.heads > 0 &&
    ctx.hiddenSize % ctx.heads === 0
  );
}

function workersStepDone(ctx: BlueprintContext): boolean {
  return ctx.jobWorkers >= 1 && ctx.jobWorkers <= 32 && ctx.jobGpus >= 1;
}

export function buildTrainingBlueprint(ctx: BlueprintContext): BlueprintStep[] {
  const steps: Omit<BlueprintStep, "waitingOn">[] = [
    {
      id: "project",
      title: "Project",
      description: "Choose which project owns this training job.",
      dependsOn: [],
      requirement: "Select or create a project.",
      volumetrics: "One project record.",
      done: ctx.selectedProjectId !== null,
    },
    {
      id: "mode",
      title: "Training mode",
      description: "Pretrain (random init) or fine-tune a Hugging Face checkpoint.",
      dependsOn: ["project"],
      requirement:
        ctx.initMode === "from_scratch"
          ? "Random init; BPE fits on corpus at job run."
          : "Load weights from the preset checkpoint.",
      volumetrics: ctx.initMode,
      done: true,
    },
    {
      id: "preset",
      title: "Model & task",
      description: "Model scale template, objective (causal LM for GPT-style), and prompt layout.",
      dependsOn: ["mode"],
      requirement: "Preset and task type selected.",
      volumetrics: "Preset template",
      done: true,
    },
    {
      id: "dataset",
      title: "Dataset",
      description: "Raw corpus — sharding and tokenization run when the job starts.",
      dependsOn: ["preset"],
      requirement: "Required fields valid for the chosen source.",
      volumetrics: ctx.dataset.source,
      done: datasetStepDone(ctx),
    },
    {
      id: "tokenizer",
      title: "Tokenizer",
      description: "Byte-pair vocabulary: train on corpus (scratch) or reuse checkpoint tokenizer.",
      dependsOn: ["dataset"],
      requirement: "Vocab and BPE settings in range.",
      volumetrics: `vocab ${(ctx.tokenizer.vocab_size ?? ctx.vocabSize).toLocaleString()}`,
      done: tokenizerStepDone(ctx),
    },
    {
      id: "geometry",
      title: "Geometry",
      description: "Width, depth, heads, and context length (defaults come from the preset).",
      dependsOn: ["tokenizer"],
      requirement: "hidden_size divisible by num_attention_heads.",
      volumetrics: `d=${ctx.hiddenSize}`,
      done: geometryStepDone(ctx),
    },
    {
      id: "diagram",
      title: "Architecture diagram",
      description: "Review the decoder stack before scheduling training steps.",
      dependsOn: ["geometry"],
      requirement: "Review the block diagram.",
      volumetrics: `L=${ctx.layers} · H=${ctx.heads}`,
      done: true,
    },
    {
      id: "training",
      title: "Schedule",
      description: "Optimization budget in gradient steps (max_steps).",
      dependsOn: ["diagram"],
      requirement: "max_steps between 10 and 500,000.",
      volumetrics: String(ctx.maxSteps),
      done: ctx.maxSteps >= 10 && ctx.maxSteps <= 500_000,
    },
    {
      id: "ray",
      title: "Ray defaults",
      description: "Optional per-project Ray settings used by distributed training.",
      dependsOn: ["training"],
      requirement: "Optional — adjust or leave defaults.",
      volumetrics: "Saved on project",
      done: true,
    },
    {
      id: "workers",
      title: "Workers & GPUs",
      description: "Ray Train worker count and GPUs per engine for this run.",
      dependsOn: ["ray"],
      requirement: "1–32 workers; ≥1 GPU per worker.",
      volumetrics: `${ctx.jobWorkers}×${ctx.jobGpus} GPU`,
      done: workersStepDone(ctx),
    },
    {
      id: "register",
      title: "Register",
      description: "Job name and OpenAI-compatible model id.",
      dependsOn: ["workers"],
      requirement: "Valid job name and model id.",
      volumetrics: ctx.openaiModelId || "—",
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
