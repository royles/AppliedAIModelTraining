import { useEffect, useMemo } from "react";
import {
  blueprintReady,
  buildTrainingBlueprint,
  type BlueprintContext,
} from "./trainingBlueprint";
import type { TrainingDatasetConfig, TokenizerBuildConfig } from "./api";
import { useDebouncedDatasetPreview } from "./hooks/useDebouncedDatasetPreview";
import { useDebouncedMinCorpusChars } from "./hooks/useDebouncedMinCorpusChars";

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
  confirmedStepIds: ReadonlySet<string>;
  selectedStepId?: string | null;
  onStepSelect?: (stepId: string) => void;
  onStepsChange?: (steps: ReturnType<typeof buildTrainingBlueprint>) => void;
  canStart: boolean;
  onStartTask: () => void;
  startBusy?: boolean;
};

export function TrainingBlueprintChecklist(props: Props) {
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
    selectedStepId,
    onStepSelect,
    onStepsChange,
    canStart,
    onStartTask,
    startBusy,
  } = props;

  const { preview, previewError, previewLoading } = useDebouncedDatasetPreview(dataset);
  const minCorpusChars = useDebouncedMinCorpusChars({
    initMode,
    tokenizer,
    vocabSize,
    hiddenSize,
    layers,
    heads,
    maxSteps,
  });

  const ctx: BlueprintContext = useMemo(
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
      datasetPreviewLoading: previewLoading,
      datasetPreviewError: previewError,
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
      previewLoading,
      previewError,
      minCorpusChars,
    ],
  );

  const steps = useMemo(() => buildTrainingBlueprint(ctx), [ctx]);
  const ready = blueprintReady(steps, confirmedStepIds, ctx);

  useEffect(() => {
    onStepsChange?.(steps);
  }, [steps, onStepsChange]);

  const doneCount = steps.filter((s) => s.done && confirmedStepIds.has(s.id)).length;

  return (
    <article className="card blueprint-checklist">
      <header className="blueprint-header">
        <h3>Task blueprint</h3>
        <span className="blueprint-progress">
          {doneCount}/{steps.length} confirmed
        </span>
      </header>
      <p className="hint">
        A blueprint is the full configuration for one training task. Steps unlock top to bottom;
        click a step to jump to its fields. Confirm each step on the right when values are final.
      </p>
      {previewLoading && dataset.source !== "builtin" && (
        <p className="hint">Dataset preview loading…</p>
      )}
      {previewError && dataset.source !== "builtin" && (
        <p className="hint blueprint-warn">Dataset preview: {previewError}</p>
      )}
      <ol className="blueprint-steps">
        {steps.map((step) => {
          const confirmed = confirmedStepIds.has(step.id);
          const complete = step.done && confirmed;
          const selected = selectedStepId === step.id;
          return (
            <li key={step.id}>
              <button
                type="button"
                className={`blueprint-step blueprint-step-button ${complete ? "done" : ""} ${
                  step.waitingOn.length ? "blocked" : ""
                } ${confirmed && !complete ? "confirmed-pending" : ""} ${
                  selected ? "active" : ""
                }`}
                aria-current={selected ? "step" : undefined}
                onClick={() => onStepSelect?.(step.id)}
              >
                <span className="blueprint-tick" aria-hidden>
                  {complete ? "✓" : confirmed ? "◐" : "○"}
                </span>
                <div className="blueprint-step-body">
                  <strong>{step.title}</strong>
                  <p className="blueprint-req">{step.requirement}</p>
                  <p className="hint blueprint-vol">{step.volumetrics}</p>
                  {step.waitingOn.length > 0 && (
                    <p className="hint blueprint-dep">
                      Waiting on: {step.waitingOn.join(", ")}
                    </p>
                  )}
                  {step.done && !confirmed && (
                    <p className="hint blueprint-confirm-hint">Ready — confirm on the right →</p>
                  )}
                </div>
              </button>
            </li>
          );
        })}
      </ol>
      <div className="blueprint-start-actions">
        <button type="button" disabled={!canStart || startBusy} onClick={onStartTask}>
          {startBusy ? "Starting…" : "Start job"}
        </button>
        {!canStart && (
          <p className="hint">Confirm every blueprint step before starting training.</p>
        )}
        {ready && canStart && (
          <p className="hint blueprint-all-ready">Blueprint complete — ready to start.</p>
        )}
      </div>
    </article>
  );
}
