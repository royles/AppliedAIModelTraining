import { useEffect, useMemo, useState } from "react";
import { api } from "./api";
import {
  buildTrainingRecipe,
  type DatasetPreview,
  type RecipeContext,
  recipeReady,
} from "./trainingRecipe";
import type { TrainingDatasetConfig, TokenizerBuildConfig } from "./api";

type Props = {
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
  selectedExp: number | null; // project id
  jobName: string;
  openaiModelId: string;
  paramEstimate: number | null;
  confirmedStepIds: ReadonlySet<string>;
  onStepSelect?: (stepId: string) => void;
  onStepsChange?: (steps: ReturnType<typeof buildTrainingRecipe>) => void;
};

export function TrainingRecipeChecklist(props: Props) {
  const {
    initMode,
    tokenizer,
    vocabSize,
    hiddenSize,
    layers,
    heads,
    maxSteps,
    jobWorkers,
    jobGpus,
    dataset,
    selectedExp,
    jobName,
    openaiModelId,
    paramEstimate,
    confirmedStepIds,
    onStepSelect,
    onStepsChange,
  } = props;

  const [preview, setPreview] = useState<DatasetPreview | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [minCorpusChars, setMinCorpusChars] = useState<number | null>(null);

  useEffect(() => {
    if (initMode !== "from_scratch") {
      setMinCorpusChars(null);
      return;
    }
    api
      .fromScratchGuidance({
        init_mode: "from_scratch",
        task_type: "causal_lm",
        architecture: {
          family: "gpt2",
          vocab_size: vocabSize,
          hidden_size: hiddenSize,
          num_hidden_layers: layers,
          num_attention_heads: heads,
          max_position_embeddings: 512,
          tie_word_embeddings: true,
        },
        tokenizer,
        max_steps: maxSteps,
        max_seq_length: 256,
        per_device_train_batch_size: 2,
        learning_rate: 6e-4,
        dataset,
      })
      .then((g) => setMinCorpusChars((g.min_corpus_characters as number) ?? null))
      .catch(() => setMinCorpusChars(null));
  }, [initMode, tokenizer, vocabSize, hiddenSize, layers, heads, maxSteps, dataset]);

  useEffect(() => {
    let cancelled = false;
    setPreviewError(null);
    api
      .datasetPreview(dataset)
      .then((p) => {
        if (!cancelled) setPreview(p);
      })
      .catch((e) => {
        if (!cancelled) {
          setPreview(null);
          setPreviewError(e instanceof Error ? e.message : String(e));
        }
      });
    return () => {
      cancelled = true;
    };
  }, [dataset]);

  const ctx: RecipeContext = useMemo(
    () => ({
      initMode,
      tokenizer,
      vocabSize,
      hiddenSize,
      layers,
      heads,
      maxSteps,
      jobWorkers,
      jobGpus,
      dataset,
      selectedProjectId: selectedExp,
      jobName,
      openaiModelId,
      paramEstimate,
      datasetPreview: preview,
      minCorpusChars,
    }),
    [
      initMode,
      tokenizer,
      vocabSize,
      hiddenSize,
      layers,
      heads,
      maxSteps,
      jobWorkers,
      jobGpus,
      dataset,
      selectedExp,
      jobName,
      openaiModelId,
      paramEstimate,
      preview,
      minCorpusChars,
    ],
  );

  const steps = useMemo(() => buildTrainingRecipe(ctx), [ctx]);
  const ready = recipeReady(steps, confirmedStepIds);

  useEffect(() => {
    onStepsChange?.(steps);
  }, [steps, onStepsChange]);

  const doneCount = steps.filter(
    (s) => s.done && confirmedStepIds.has(s.id),
  ).length;

  return (
    <article className="card recipe-checklist">
      <header className="recipe-header">
        <h3>Training recipe</h3>
        <span className="recipe-progress">
          {doneCount}/{steps.length} confirmed
        </span>
      </header>
      <p className="hint">
        Steps unlock top to bottom. Click a step to jump to its config. Use{" "}
        <strong>Confirm step</strong> on the right when values are final.
      </p>
      {previewError && dataset.source !== "builtin" && (
        <p className="hint recipe-warn">Dataset preview: {previewError}</p>
      )}
      <ol className="recipe-steps">
        {steps.map((step) => {
          const confirmed = confirmedStepIds.has(step.id);
          const complete = step.done && confirmed;
          return (
            <li key={step.id}>
              <button
                type="button"
                className={`recipe-step recipe-step-button ${complete ? "done" : ""} ${
                  step.waitingOn.length ? "blocked" : ""
                } ${confirmed && !complete ? "confirmed-pending" : ""}`}
                onClick={() => onStepSelect?.(step.id)}
              >
                <span className="recipe-tick" aria-hidden>
                  {complete ? "✓" : confirmed ? "◐" : "○"}
                </span>
                <div className="recipe-step-body">
                  <strong>{step.title}</strong>
                  <p className="recipe-req">{step.requirement}</p>
                  <p className="hint recipe-vol">{step.volumetrics}</p>
                  {step.waitingOn.length > 0 && (
                    <p className="hint recipe-dep">
                      Waiting on: {step.waitingOn.join(", ")}
                    </p>
                  )}
                  {step.done && !confirmed && (
                    <p className="hint recipe-confirm-hint">Ready — confirm on the right →</p>
                  )}
                </div>
              </button>
            </li>
          );
        })}
      </ol>
      {ready && <p className="hint recipe-all-ready">All steps confirmed — you can start the task.</p>}
    </article>
  );
}
