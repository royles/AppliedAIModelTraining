import type { Project, TrainingDatasetConfig, TransformerPreset, TokenizerBuildConfig } from "./api";
import { DatasetConfigFields } from "./DatasetConfigFields";
import { FromScratchConfigFields } from "./FromScratchConfigFields";
import { RecipeSection } from "./RecipeSection";
import { TrainingRecipeChecklist } from "./TrainingRecipeChecklist";
import { TransformerArchitectureDiagram } from "./TransformerArchitectureDiagram";
import { recipeReady, type RecipeStep } from "./trainingRecipe";
import { useCallback, useMemo, useState } from "react";

type Props = {
  presets: TransformerPreset[];
  preset: string;
  onPresetChange: (name: string) => void;
  selectedPreset: TransformerPreset | undefined;
  hiddenSize: number;
  layers: number;
  heads: number;
  maxSteps: number;
  paramEstimate: number | null;
  onHiddenSizeChange: (n: number) => void;
  onLayersChange: (n: number) => void;
  onHeadsChange: (n: number) => void;
  onMaxStepsChange: (n: number) => void;
  projects: Project[];
  selectedProjectId: number | null;
  onSelectedProjectIdChange: (id: number) => void;
  projectName: string;
  onProjectNameChange: (v: string) => void;
  onCreateProject: () => void;
  jobName: string;
  onJobNameChange: (v: string) => void;
  openaiModelId: string;
  onOpenaiModelIdChange: (v: string) => void;
  jobWorkers: number;
  onJobWorkersChange: (n: number) => void;
  jobGpus: number;
  onJobGpusChange: (n: number) => void;
  onCreateJob: () => void;
  dataset: TrainingDatasetConfig;
  onDatasetChange: (d: TrainingDatasetConfig) => void;
  initMode: "from_scratch" | "pretrained";
  onInitModeChange: (m: "from_scratch" | "pretrained") => void;
  tokenizer: TokenizerBuildConfig;
  onTokenizerChange: (t: TokenizerBuildConfig) => void;
  vocabSize: number;
  onVocabSizeChange: (n: number) => void;
  maxPosition: number;
  onMaxPositionChange: (n: number) => void;
  archFamily: "gpt2" | "bert";
  taskType: "causal_lm" | "sequence_classification";
  intermediateSize: number;
  tieWordEmbeddings: boolean;
  baseModel?: string | null;
  useLora?: boolean;
};

const STEP_ORDER = [
  "project",
  "mode",
  "dataset",
  "tokenizer",
  "architecture",
  "training",
  "compute",
  "register",
] as const;

function dependentsOf(stepId: string): string[] {
  const idx = STEP_ORDER.indexOf(stepId as (typeof STEP_ORDER)[number]);
  if (idx < 0) return [];
  return STEP_ORDER.slice(idx + 1);
}

export function ModelTasksPanel(props: Props) {
  const {
    presets,
    preset,
    onPresetChange,
    selectedPreset,
    hiddenSize,
    layers,
    heads,
    maxSteps,
    paramEstimate,
    onHiddenSizeChange,
    onLayersChange,
    onHeadsChange,
    onMaxStepsChange,
    projects,
    selectedProjectId,
    onSelectedProjectIdChange,
    projectName,
    onProjectNameChange,
    onCreateProject,
    jobName,
    onJobNameChange,
    openaiModelId,
    onOpenaiModelIdChange,
    jobWorkers,
    onJobWorkersChange,
    jobGpus,
    onJobGpusChange,
    onCreateJob,
    dataset,
    onDatasetChange,
    initMode,
    onInitModeChange,
    tokenizer,
    onTokenizerChange,
    vocabSize,
    onVocabSizeChange,
    maxPosition,
    onMaxPositionChange,
    archFamily,
    taskType,
    intermediateSize,
    tieWordEmbeddings,
    baseModel,
    useLora,
  } = props;

  const [confirmedSteps, setConfirmedSteps] = useState<Set<string>>(() => new Set());
  const [recipeSteps, setRecipeSteps] = useState<RecipeStep[]>([]);

  const invalidateFrom = useCallback((stepId: string) => {
    setConfirmedSteps((prev) => {
      const next = new Set(prev);
      next.delete(stepId);
      for (const id of dependentsOf(stepId)) {
        next.delete(id);
      }
      return next;
    });
  }, []);

  const confirmStep = useCallback((stepId: string) => {
    setConfirmedSteps((prev) => new Set(prev).add(stepId));
  }, []);

  const scrollToStep = useCallback((stepId: string) => {
    document.getElementById(`recipe-step-${stepId}`)?.scrollIntoView({
      behavior: "smooth",
      block: "start",
    });
  }, []);

  const ready = recipeReady(recipeSteps, confirmedSteps);

  const stepById = useMemo(
    () => new Map(recipeSteps.map((s) => [s.id, s])),
    [recipeSteps],
  );

  return (
    <div className="settings-page model-tasks-page">
      <header className="main-panel-header">
        <h2>Model Tasks</h2>
        <p className="hint">
          Confirm each recipe step on the right after configuring it. Start training when every step
          is checked on the left.
        </p>
      </header>

      <div className="model-tasks-recipe-layout">
        <aside className="recipe-layout-sidebar">
          <TrainingRecipeChecklist
            initMode={initMode}
            tokenizer={tokenizer}
            vocabSize={vocabSize}
            hiddenSize={hiddenSize}
            layers={layers}
            heads={heads}
            maxSteps={maxSteps}
            jobWorkers={jobWorkers}
            jobGpus={jobGpus}
            dataset={dataset}
            selectedExp={selectedProjectId}
            jobName={jobName}
            openaiModelId={openaiModelId}
            paramEstimate={paramEstimate}
            confirmedStepIds={confirmedSteps}
            onStepSelect={scrollToStep}
            onStepsChange={setRecipeSteps}
          />
        </aside>

        <div className="recipe-layout-config">
          <RecipeSection
            stepId="project"
            title="Project"
            canConfirm={stepById.get("project")?.done ?? false}
            confirmed={confirmedSteps.has("project")}
            confirmHint="Create or select a project first."
            onConfirm={() => confirmStep("project")}
          >
            <label>
              New project name
              <input
                value={projectName}
                onChange={(e) => {
                  invalidateFrom("project");
                  onProjectNameChange(e.target.value);
                }}
              />
            </label>
            <button type="button" onClick={onCreateProject}>Create project</button>
            <label>
              Active project for new tasks
              <select
                value={selectedProjectId ?? ""}
                onChange={(e) => {
                  invalidateFrom("project");
                  onSelectedProjectIdChange(Number(e.target.value));
                }}
              >
                {projects.map((p) => (
                  <option key={p.id} value={p.id}>{p.name}</option>
                ))}
              </select>
            </label>
          </RecipeSection>

          <RecipeSection
            stepId="mode"
            title="Training mode"
            canConfirm={stepById.get("mode")?.done ?? false}
            confirmed={confirmedSteps.has("mode")}
            onConfirm={() => confirmStep("mode")}
          >
            <FromScratchConfigFields
              section="mode"
              initMode={initMode}
              onInitModeChange={(m) => {
                invalidateFrom("mode");
                onInitModeChange(m);
              }}
              tokenizer={tokenizer}
              onTokenizerChange={(t) => {
                invalidateFrom("tokenizer");
                onTokenizerChange(t);
              }}
              vocabSize={vocabSize}
              onVocabSizeChange={(n) => {
                invalidateFrom("tokenizer");
                onVocabSizeChange(n);
              }}
              maxSteps={maxSteps}
              hiddenSize={hiddenSize}
              layers={layers}
            />
          </RecipeSection>

          <RecipeSection
            stepId="dataset"
            title="Training dataset"
            canConfirm={stepById.get("dataset")?.done ?? false}
            confirmed={confirmedSteps.has("dataset")}
            confirmHint="Fix dataset source and wait for preview if required."
            onConfirm={() => confirmStep("dataset")}
          >
            <DatasetConfigFields
              value={dataset}
              onChange={(d) => {
                invalidateFrom("dataset");
                onDatasetChange(d);
              }}
            />
          </RecipeSection>

          <RecipeSection
            stepId="tokenizer"
            title="Tokenizer & vocabulary"
            canConfirm={stepById.get("tokenizer")?.done ?? false}
            confirmed={confirmedSteps.has("tokenizer")}
            confirmHint="Match vocab to corpus size and architecture."
            onConfirm={() => confirmStep("tokenizer")}
          >
            <FromScratchConfigFields
              section="tokenizer"
              initMode={initMode}
              onInitModeChange={onInitModeChange}
              tokenizer={tokenizer}
              onTokenizerChange={(t) => {
                invalidateFrom("tokenizer");
                onTokenizerChange(t);
              }}
              vocabSize={vocabSize}
              onVocabSizeChange={(n) => {
                invalidateFrom("tokenizer");
                onVocabSizeChange(n);
              }}
              maxSteps={maxSteps}
              hiddenSize={hiddenSize}
              layers={layers}
            />
          </RecipeSection>

          <RecipeSection
            stepId="architecture"
            title="Transformer geometry"
            canConfirm={stepById.get("architecture")?.done ?? false}
            confirmed={confirmedSteps.has("architecture")}
            confirmHint="hidden_size must divide evenly by attention heads."
            onConfirm={() => confirmStep("architecture")}
          >
            <label>
              Preset
              <select
                value={preset}
                onChange={(e) => {
                  invalidateFrom("architecture");
                  onPresetChange(e.target.value);
                }}
              >
                {presets.map((p) => (
                  <option key={p.name} value={p.name}>{p.title}</option>
                ))}
              </select>
            </label>
            <TransformerArchitectureDiagram
              presetTitle={selectedPreset?.title}
              family={archFamily}
              initMode={initMode}
              taskType={taskType}
              baseModel={baseModel}
              useLora={useLora}
              hiddenSize={hiddenSize}
              layers={layers}
              heads={heads}
              vocabSize={vocabSize}
              maxPosition={maxPosition}
              intermediateSize={intermediateSize}
              tieWordEmbeddings={tieWordEmbeddings}
              paramEstimate={paramEstimate}
              onHiddenSizeChange={(n) => {
                invalidateFrom("architecture");
                onHiddenSizeChange(n);
              }}
              onLayersChange={(n) => {
                invalidateFrom("architecture");
                onLayersChange(n);
              }}
              onHeadsChange={(n) => {
                invalidateFrom("architecture");
                onHeadsChange(n);
              }}
            />
            <div className="arch-grid arch-grid-compact">
              <label>
                Hidden size
                <input
                  type="number"
                  value={hiddenSize}
                  onChange={(e) => {
                    invalidateFrom("architecture");
                    onHiddenSizeChange(Number(e.target.value));
                  }}
                />
              </label>
              <label>
                Layers
                <input
                  type="number"
                  value={layers}
                  onChange={(e) => {
                    invalidateFrom("architecture");
                    onLayersChange(Number(e.target.value));
                  }}
                />
              </label>
              <label>
                Attention heads
                <input
                  type="number"
                  value={heads}
                  onChange={(e) => {
                    invalidateFrom("architecture");
                    onHeadsChange(Number(e.target.value));
                  }}
                />
              </label>
              <label>
                Max positions
                <input
                  type="number"
                  min={128}
                  value={maxPosition}
                  onChange={(e) => {
                    invalidateFrom("architecture");
                    onMaxPositionChange(Number(e.target.value));
                  }}
                />
              </label>
            </div>
          </RecipeSection>

          <RecipeSection
            stepId="training"
            title="Optimization schedule"
            canConfirm={stepById.get("training")?.done ?? false}
            confirmed={confirmedSteps.has("training")}
            confirmHint="Set max_steps between 10 and 500,000."
            onConfirm={() => confirmStep("training")}
          >
            <label>
              Training steps (max_steps)
              <input
                type="number"
                value={maxSteps}
                onChange={(e) => {
                  invalidateFrom("training");
                  onMaxStepsChange(Number(e.target.value));
                }}
              />
            </label>
          </RecipeSection>

          <RecipeSection
            stepId="compute"
            title="Ray / CDSW compute"
            canConfirm={stepById.get("compute")?.done ?? false}
            confirmed={confirmedSteps.has("compute")}
            onConfirm={() => confirmStep("compute")}
          >
            <div className="arch-grid">
              <label>
                Ray / CDSW workers for new jobs
                <input
                  type="number"
                  min={1}
                  max={32}
                  value={jobWorkers}
                  onChange={(e) => {
                    invalidateFrom("compute");
                    onJobWorkersChange(Number(e.target.value));
                  }}
                />
              </label>
              <label>
                GPUs per worker
                <input
                  type="number"
                  min={0}
                  max={8}
                  value={jobGpus}
                  onChange={(e) => {
                    invalidateFrom("compute");
                    onJobGpusChange(Number(e.target.value));
                  }}
                />
              </label>
            </div>
          </RecipeSection>

          <RecipeSection
            stepId="register"
            title="Task metadata & /v1 id"
            canConfirm={stepById.get("register")?.done ?? false}
            confirmed={confirmedSteps.has("register")}
            confirmHint="Job name and OpenAI model id (letters, digits, . _ -)."
            onConfirm={() => confirmStep("register")}
          >
            <label>
              OpenAI model id
              <input
                value={openaiModelId}
                onChange={(e) => {
                  invalidateFrom("register");
                  onOpenaiModelIdChange(e.target.value);
                }}
              />
            </label>
            <label>
              Job name
              <input
                value={jobName}
                onChange={(e) => {
                  invalidateFrom("register");
                  onJobNameChange(e.target.value);
                }}
              />
            </label>
            <button type="button" onClick={onCreateJob} disabled={!ready}>
              Start model task
            </button>
            {!ready && (
              <p className="hint">
                Confirm every recipe step on the left (green checks) before starting.
              </p>
            )}
          </RecipeSection>
        </div>
      </div>
    </div>
  );
}
