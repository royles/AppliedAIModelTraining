import type { TrainingDatasetConfig, TokenizerBuildConfig } from "./api";

export type DatasetPreview = {
  examples: number;
  total_characters: number;
  source: string;
};

export type RecipeContext = {
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
  selectedExp: number | null;
  jobName: string;
  openaiModelId: string;
  paramEstimate: number | null;
  datasetPreview: DatasetPreview | null;
  minCorpusChars: number | null;
};

export type RecipeStep = {
  id: string;
  title: string;
  dependsOn: string[];
  requirement: string;
  volumetrics: string;
  done: boolean;
  waitingOn: string[];
};

const OPENAI_ID = /^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$/;

function corpusChars(ctx: RecipeContext): number {
  if (!ctx.datasetPreview) return 0;
  return ctx.datasetPreview.total_characters;
}

export function buildTrainingRecipe(ctx: RecipeContext): RecipeStep[] {
  const steps: Omit<RecipeStep, "waitingOn">[] = [
    {
      id: "experiment",
      title: "Experiment",
      dependsOn: [],
      requirement: "Select an experiment to group runs (create one if empty).",
      volumetrics: "1 experiment record in SQLite.",
      done: ctx.selectedExp !== null,
    },
    {
      id: "mode",
      title: "Training mode",
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
      dependsOn: ["mode"],
      requirement:
        ctx.initMode === "from_scratch" && ctx.tokenizer.mode === "train_bpe_from_dataset"
          ? "Non-demo corpus: inline text, data/datasets file, or Hugging Face split."
          : "Any configured source (builtin OK for smoke tests).",
      volumetrics: ctx.datasetPreview
        ? `${ctx.datasetPreview.examples.toLocaleString()} examples · ${corpusChars(ctx).toLocaleString()} characters (after repeat)`
        : "Load preview to measure examples and characters.",
      done: (() => {
        if (ctx.initMode === "from_scratch" && ctx.tokenizer.mode === "train_bpe_from_dataset") {
          if (ctx.dataset.source === "builtin") return false;
          if (!ctx.datasetPreview || ctx.datasetPreview.examples < 1) return false;
          const min = ctx.minCorpusChars ?? 10_000;
          return corpusChars(ctx) >= min;
        }
        if (ctx.dataset.source === "inline") {
          return (ctx.dataset.texts?.length ?? 0) > 0;
        }
        if (ctx.dataset.source === "builtin") return true;
        return Boolean(ctx.dataset.path || ctx.dataset.hf_dataset);
      })(),
    },
    {
      id: "tokenizer",
      title: "Tokenizer & vocabulary",
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
        return vocabOk && corpusChars(ctx) >= min && ctx.dataset.source !== "builtin";
      })(),
    },
    {
      id: "architecture",
      title: "Transformer geometry",
      dependsOn: ["tokenizer"],
      requirement: "hidden_size divisible by num_attention_heads; GPT-2 family for scratch causal LM.",
      volumetrics: `d=${ctx.hiddenSize} · L=${ctx.layers} · H=${ctx.heads} · ~${
        ctx.paramEstimate?.toLocaleString() ?? "?"
      } params`,
      done:
        ctx.hiddenSize > 0 &&
        ctx.layers > 0 &&
        ctx.heads > 0 &&
        ctx.hiddenSize % ctx.heads === 0 &&
        ctx.paramEstimate !== null &&
        ctx.paramEstimate < 500_000_000,
    },
    {
      id: "training",
      title: "Optimization schedule",
      dependsOn: ["architecture"],
      requirement: "max_steps and learning rate set; logging_steps implied by preset.",
      volumetrics: `max_steps ${ctx.maxSteps.toLocaleString()} (typical scratch smoke: 40–500 · serious: 5k–50k+)`,
      done: ctx.maxSteps >= 10 && ctx.maxSteps <= 500_000,
    },
    {
      id: "compute",
      title: "Ray / CDSW compute",
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
      title: "Task metadata & /v1 id",
      dependsOn: ["compute"],
      requirement: "Job name and OpenAI-compatible model id for registration after success.",
      volumetrics: "model id 1–64 chars: letters, digits, . _ -",
      done:
        ctx.jobName.trim().length > 0 && OPENAI_ID.test(ctx.openaiModelId.trim()),
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

export function recipeReady(steps: RecipeStep[]): boolean {
  return steps.length > 0 && steps.every((s) => s.done);
}
