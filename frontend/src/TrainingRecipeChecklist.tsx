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
  selectedExp: number | null;
  jobName: string;
  openaiModelId: string;
  paramEstimate: number | null;
  onReadyChange?: (ready: boolean) => void;
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
    onReadyChange,
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
      selectedExp,
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
  const ready = recipeReady(steps);

  useEffect(() => {
    onReadyChange?.(ready);
  }, [ready, onReadyChange]);

  const doneCount = steps.filter((s) => s.done).length;

  return (
    <article className="card recipe-checklist">
      <header className="recipe-header">
        <h3>Training recipe</h3>
        <span className="recipe-progress">
          {doneCount}/{steps.length} complete
        </span>
      </header>
      <p className="hint">
        Hard dependencies run top to bottom. Each step shows required volumetrics; a tick means the
        current configuration satisfies it.
      </p>
      {previewError && dataset.source !== "builtin" && (
        <p className="hint recipe-warn">Dataset preview: {previewError}</p>
      )}
      <ol className="recipe-steps">
        {steps.map((step) => (
          <li
            key={step.id}
            className={`recipe-step ${step.done ? "done" : ""} ${step.waitingOn.length ? "blocked" : ""}`}
          >
            <span className="recipe-tick" aria-hidden>{step.done ? "✓" : "○"}</span>
            <div className="recipe-step-body">
              <strong>{step.title}</strong>
              <p className="recipe-req">{step.requirement}</p>
              <p className="hint recipe-vol">{step.volumetrics}</p>
              {step.waitingOn.length > 0 && (
                <p className="hint recipe-dep">
                  Waiting on: {step.waitingOn.join(", ")}
                </p>
              )}
            </div>
          </li>
        ))}
      </ol>
    </article>
  );
}
