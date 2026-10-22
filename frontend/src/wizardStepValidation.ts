import type { BlueprintContext } from "./trainingBlueprint";
import { buildTrainingBlueprint } from "./trainingBlueprint";
import { validateDatasetConfig } from "./datasetValidation";
import type { WizardValidationMessage } from "./wizardValidationMessage";

const OPENAI_ID = /^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$/;

function block(message: string, fix: string): WizardValidationMessage {
  return { message, fix };
}

function divisorsUpTo(n: number, max: number): number[] {
  const out: number[] = [];
  for (let d = 1; d <= Math.min(max, n); d++) {
    if (n % d === 0) {
      out.push(d);
    }
  }
  return out;
}

function suggestHeadCount(hiddenSize: number, currentHeads: number): string {
  const valid = divisorsUpTo(hiddenSize, 128);
  if (valid.length === 0) {
    return "Increase hidden size or lower head count so hidden size divides evenly by heads.";
  }
  let best = valid[0];
  let bestDist = Math.abs(best - currentHeads);
  for (const h of valid) {
    const dist = Math.abs(h - currentHeads);
    if (dist < bestDist) {
      best = h;
      bestDist = dist;
    }
  }
  return `Set attention heads to ${best} (hidden ${hiddenSize} ÷ ${best} = ${hiddenSize / best}), or pick another divisor of ${hiddenSize}.`;
}

/**
 * Instant, client-only block for the current wizard step.
 * No API calls and no dataset loading.
 */
export function wizardStepBlockMessage(
  stepId: string,
  ctx: BlueprintContext,
): WizardValidationMessage | null {
  const steps = buildTrainingBlueprint(ctx);
  const step = steps.find((s) => s.id === stepId);
  if (!step || step.done) {
    return null;
  }

  switch (stepId) {
    case "project":
      return block(
        "No project is selected for this job.",
        "Create a project on this step or pick one from the list.",
      );
    case "dataset": {
      const v = validateDatasetConfig(ctx.dataset);
      if (!v.valid && v.block) {
        return v.block;
      }
      if (
        ctx.initMode === "from_scratch" &&
        ctx.tokenizer.mode === "train_bpe_from_dataset" &&
        ctx.dataset.source === "builtin"
      ) {
        return block(
          "Built-in demo data cannot train a from-scratch BPE tokenizer.",
          "Switch source to Hugging Face, a file under data/datasets/, inline text, Iceberg, or S3 Parquet.",
        );
      }
      return block(step.requirement, "Review the fields on this step and correct any highlighted issues.");
    }
    case "tokenizer": {
      if (ctx.initMode !== "from_scratch") {
        return block(step.requirement, "Fine-tuning uses the preset tokenizer; no changes needed here.");
      }
      const datasetCheck = validateDatasetConfig(ctx.dataset);
      if (!datasetCheck.valid && datasetCheck.block) {
        return {
          message: datasetCheck.block.message,
          fix: `${datasetCheck.block.fix} (Dataset step.)`,
        };
      }
      if (
        ctx.tokenizer.mode === "train_bpe_from_dataset" &&
        ctx.dataset.source === "builtin"
      ) {
        return block(
          "Built-in demo data cannot train BPE.",
          "Go to Dataset and choose a real corpus; loading happens when the job runs.",
        );
      }
      if (ctx.tokenizer.mode === "pretrained_gpt2" && ctx.vocabSize < 50257) {
        return block(
          "GPT-2 vocabulary mode needs the full GPT-2 vocab.",
          "Set target vocab to 50257 or switch tokenizer mode to Train BPE on dataset.",
        );
      }
      if (ctx.tokenizer.mode === "train_bpe_from_dataset") {
        const tokVocab = ctx.tokenizer.vocab_size ?? ctx.vocabSize;
        if (tokVocab < 256 || tokVocab > 256_000) {
          return block(
            `Target vocab ${tokVocab} is out of range.`,
            "Use a value between 256 and 256,000 (2048–8192 is typical for small scratch models).",
          );
        }
        if ((ctx.tokenizer.min_frequency ?? 2) < 1) {
          return block(
            "BPE min frequency must be at least 1.",
            "Leave min frequency at 2 unless you know you need a rarer-token threshold.",
          );
        }
      }
      return block(step.requirement, "Check vocab size and BPE options above.");
    }
    case "architecture":
      if (ctx.hiddenSize <= 0 || ctx.layers <= 0 || ctx.heads <= 0) {
        return block(
          "Hidden size, layer count, and head count must be positive.",
          "Enter values on this step (the preset fills them for fine-tunes).",
        );
      }
      if (ctx.hiddenSize % ctx.heads !== 0) {
        return block(
          `Hidden size ${ctx.hiddenSize} is not divisible by ${ctx.heads} attention heads.`,
          suggestHeadCount(ctx.hiddenSize, ctx.heads),
        );
      }
      return block(step.requirement, "Adjust geometry until the diagram and metrics look correct.");
    case "training":
      if (ctx.maxSteps < 10 || ctx.maxSteps > 500_000) {
        return block(
          `max_steps is ${ctx.maxSteps}; it must be between 10 and 500,000.`,
          "Use 40–500 for smoke tests or thousands+ for serious training.",
        );
      }
      return block(step.requirement, "Set max_steps on the Schedule step.");
    case "compute":
      if (ctx.jobWorkers < 1 || ctx.jobWorkers > 32) {
        return block(
          `Worker count is ${ctx.jobWorkers}; allowed range is 1–32.`,
          "Set Ray/CDSW worker engines to match your cluster quota.",
        );
      }
      if (ctx.jobGpus < 1) {
        return block(
          "GPUs per worker must be at least 1 for this configuration.",
          "Set GPUs per engine to 1+, or use a single CPU worker for tiny models.",
        );
      }
      return block(step.requirement, "Review job engine sizing on the Compute step.");
    case "register":
      if (!ctx.jobName.trim()) {
        return block("Job name is empty.", "Enter a short label you will recognize in the job list.");
      }
      if (!OPENAI_ID.test(ctx.openaiModelId.trim())) {
        return block(
          "OpenAI model id format is invalid.",
          "Use 1–64 characters: start with a letter or digit, then letters, digits, dots, underscores, or hyphens (e.g. my-model-v1).",
        );
      }
      return block(step.requirement, "Fill job name and model id on this step.");
    case "mode":
    case "preset":
    default:
      return block(step.requirement, "Complete the required fields on this step.");
  }
}

/** @deprecated Use wizardStepBlockMessage and read .message / .fix */
export function wizardStepBlockReason(stepId: string, ctx: BlueprintContext): string | null {
  const msg = wizardStepBlockMessage(stepId, ctx);
  if (!msg) {
    return null;
  }
  return `${msg.message} Suggestion: ${msg.fix}`;
}

const WIZARD_STEP_ORDER = [
  "project",
  "mode",
  "preset",
  "dataset",
  "tokenizer",
  "architecture",
  "training",
  "compute",
  "register",
] as const;

export function wizardFirstBlockMessage(ctx: BlueprintContext): WizardValidationMessage | null {
  for (const stepId of WIZARD_STEP_ORDER) {
    const msg = wizardStepBlockMessage(stepId, ctx);
    if (msg) {
      return msg;
    }
  }
  return null;
}

/** @deprecated Use wizardFirstBlockMessage */
export function wizardFirstBlockReason(ctx: BlueprintContext): string | null {
  const msg = wizardFirstBlockMessage(ctx);
  return msg ? `${msg.message} Suggestion: ${msg.fix}` : null;
}
