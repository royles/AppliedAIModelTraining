import type { Project, TrainingDatasetConfig, TrainingJob, TransformerPreset, TokenizerBuildConfig } from "./api";
import { BlueprintSection } from "./BlueprintSection";
import { DatasetConfigFields } from "./DatasetConfigFields";
import { FromScratchConfigFields } from "./FromScratchConfigFields";
import { ProjectTasksFields } from "./ProjectTasksFields";
import { TrainingBlueprintChecklist } from "./TrainingBlueprintChecklist";
import { TransformerArchitectureDiagram } from "./TransformerArchitectureDiagram";
import { ValidatedNumberField } from "./ValidatedNumberField";
import { blueprintReady, buildTrainingBlueprint, type BlueprintStep } from "./trainingBlueprint";
import { useCallback, useMemo, useState, type ReactNode } from "react";

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
  jobs: TrainingJob[];
  selectedProjectId: number | null;
  onSelectedProjectIdChange: (id: number) => void;
  newProjectName: string;
  onNewProjectNameChange: (v: string) => void;
  newProjectDescription: string;
  onNewProjectDescriptionChange: (v: string) => void;
  onCreateProject: () => void;
  onUpdateProject: (projectId: number, name: string) => void;
  onDeleteProject: (projectId: number) => void;
  onUpdateJobName: (jobId: number, name: string) => void;
  onDeleteJob: (jobId: number) => void;
  projectBusy: boolean;
  jobBusy: boolean;
  jobName: string;
  onJobNameChange: (v: string) => void;
  openaiModelId: string;
  onOpenaiModelIdChange: (v: string) => void;
  jobWorkers: number;
  onJobWorkersChange: (n: number) => void;
  jobGpus: number;
  onJobGpusChange: (n: number) => void;
  onCreateJob: () => void;
  startBusy: boolean;
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
    jobs,
    selectedProjectId,
    onSelectedProjectIdChange,
    newProjectName,
    onNewProjectNameChange,
    newProjectDescription,
    onNewProjectDescriptionChange,
    onCreateProject,
    onUpdateProject,
    onDeleteProject,
    onUpdateJobName,
    onDeleteJob,
    projectBusy,
    jobBusy,
    jobName,
    onJobNameChange,
    openaiModelId,
    onOpenaiModelIdChange,
    jobWorkers,
    onJobWorkersChange,
    jobGpus,
    onJobGpusChange,
    onCreateJob,
    startBusy,
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
  const [blueprintSteps, setBlueprintSteps] = useState<BlueprintStep[]>([]);

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
    document.getElementById(`blueprint-step-${stepId}`)?.scrollIntoView({
      behavior: "smooth",
      block: "start",
    });
  }, []);

  const ready = blueprintReady(blueprintSteps, confirmedSteps);

  const stepById = useMemo(
    () => new Map(blueprintSteps.map((s) => [s.id, s])),
    [blueprintSteps],
  );

  const stepDescriptions = useMemo(() => {
    const stub = buildTrainingBlueprint({
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
      selectedProjectId,
      jobName,
      openaiModelId,
      paramEstimate: null,
      datasetPreview: null,
      minCorpusChars: null,
    });
    return new Map(stub.map((s) => [s.id, s.description]));
  }, [
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
    selectedProjectId,
    jobName,
    openaiModelId,
  ]);

  function section(
    stepId: string,
    title: string,
    canConfirm: boolean,
    confirmHint: string | undefined,
    children: ReactNode,
  ) {
    return (
      <BlueprintSection
        stepId={stepId}
        title={title}
        description={stepDescriptions.get(stepId)}
        canConfirm={canConfirm}
        confirmed={confirmedSteps.has(stepId)}
        confirmHint={confirmHint}
        onConfirm={() => confirmStep(stepId)}
      >
        {children}
      </BlueprintSection>
    );
  }

  return (
    <div className="settings-page model-tasks-page">
      <header className="main-panel-header">
        <h2>Model Tasks</h2>
        <p className="hint">
          Build a task blueprint step by step. Confirm each section on the right; when the checklist
          is complete, start training from the left.
        </p>
      </header>

      <div className="model-tasks-blueprint-layout">
        <aside className="blueprint-layout-sidebar">
          <TrainingBlueprintChecklist
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
            onStepsChange={setBlueprintSteps}
            canStart={ready}
            onStartTask={onCreateJob}
            startBusy={startBusy}
          />
        </aside>

        <div className="blueprint-layout-config">
          {section(
            "project",
            "Project & tasks",
            stepById.get("project")?.done ?? false,
            "Create or select a project first.",
            <ProjectTasksFields
              projects={projects}
              jobs={jobs}
              selectedProjectId={selectedProjectId}
              onSelectedProjectIdChange={onSelectedProjectIdChange}
              newProjectName={newProjectName}
              onNewProjectNameChange={onNewProjectNameChange}
              newProjectDescription={newProjectDescription}
              onNewProjectDescriptionChange={onNewProjectDescriptionChange}
              onCreateProject={onCreateProject}
              onUpdateProject={onUpdateProject}
              onDeleteProject={onDeleteProject}
              onUpdateJobName={onUpdateJobName}
              onDeleteJob={onDeleteJob}
              projectBusy={projectBusy}
              jobBusy={jobBusy}
              onInvalidateBlueprint={() => invalidateFrom("project")}
            />,
          )}

          {section(
            "mode",
            "Training mode",
            stepById.get("mode")?.done ?? false,
            undefined,
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
            />,
          )}

          {section(
            "dataset",
            "Training dataset",
            stepById.get("dataset")?.done ?? false,
            "Fix dataset source and wait for preview if required.",
            <DatasetConfigFields
              value={dataset}
              onChange={(d) => {
                invalidateFrom("dataset");
                onDatasetChange(d);
              }}
            />,
          )}

          {section(
            "tokenizer",
            "Tokenizer & vocabulary",
            stepById.get("tokenizer")?.done ?? false,
            "Match vocab to corpus size and architecture.",
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
            />,
          )}

          {section(
            "architecture",
            "Transformer geometry",
            stepById.get("architecture")?.done ?? false,
            "hidden_size must divide evenly by attention heads.",
            <>
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
                  <ValidatedNumberField
                    value={hiddenSize}
                    min={1}
                    max={8192}
                    onChange={(n) => {
                      invalidateFrom("architecture");
                      onHiddenSizeChange(n);
                    }}
                  />
                </label>
                <label>
                  Layers
                  <ValidatedNumberField
                    value={layers}
                    min={1}
                    max={128}
                    onChange={(n) => {
                      invalidateFrom("architecture");
                      onLayersChange(n);
                    }}
                  />
                </label>
                <label>
                  Attention heads
                  <ValidatedNumberField
                    value={heads}
                    min={1}
                    max={128}
                    onChange={(n) => {
                      invalidateFrom("architecture");
                      onHeadsChange(n);
                    }}
                  />
                </label>
                <label>
                  Max positions
                  <ValidatedNumberField
                    value={maxPosition}
                    min={128}
                    max={32768}
                    onChange={(n) => {
                      invalidateFrom("architecture");
                      onMaxPositionChange(n);
                    }}
                  />
                </label>
              </div>
            </>,
          )}

          {section(
            "training",
            "Optimization schedule",
            stepById.get("training")?.done ?? false,
            "Set max_steps between 10 and 500,000.",
            <label>
              Training steps (max_steps)
              <ValidatedNumberField
                value={maxSteps}
                min={10}
                max={500_000}
                onChange={(n) => {
                  invalidateFrom("training");
                  onMaxStepsChange(n);
                }}
              />
            </label>,
          )}

          {section(
            "compute",
            "Ray / CDSW compute",
            stepById.get("compute")?.done ?? false,
            undefined,
            <div className="arch-grid">
              <label>
                Ray / CDSW workers for new jobs
                <ValidatedNumberField
                  value={jobWorkers}
                  min={1}
                  max={32}
                  onChange={(n) => {
                    invalidateFrom("compute");
                    onJobWorkersChange(n);
                  }}
                />
              </label>
              <label>
                GPUs per worker
                <ValidatedNumberField
                  value={jobGpus}
                  min={0}
                  max={8}
                  onChange={(n) => {
                    invalidateFrom("compute");
                    onJobGpusChange(n);
                  }}
                />
              </label>
            </div>,
          )}

          {section(
            "register",
            "Task metadata & /v1 id",
            stepById.get("register")?.done ?? false,
            "Job name and OpenAI model id (letters, digits, . _ -).",
            <>
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
            </>,
          )}
        </div>
      </div>
    </div>
  );
}
