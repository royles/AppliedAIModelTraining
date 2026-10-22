import type {
  Project,
  PromptFormatSetting,
  TaskDefinition,
  TaskVersion,
  TrainingDatasetConfig,
  TrainingJob,
  TransformerPreset,
  TokenizerBuildConfig,
} from "./api";
import { api } from "./api";
import { DatasetConfigFields } from "./DatasetConfigFields";
import { FromScratchConfigFields } from "./FromScratchConfigFields";
import { ModelTaskWizardStepper } from "./ModelTaskWizardStepper";
import { PlatformSettingsPanel } from "./PlatformSettingsPanel";
import { ValidatedNumberField } from "./ValidatedNumberField";
import { ArchitectureGeometryStep } from "./components/ArchitectureGeometryStep";
import { PresetWizardStep } from "./components/PresetWizardStep";
import { ProjectWizardStep } from "./components/ProjectWizardStep";
import {
  buildModelTaskBlueprint,
  downloadBlueprintJson,
  parseModelTaskBlueprintFile,
  type ModelTaskBlueprintFile,
  type ModelTaskWizardState,
} from "./modelTaskBlueprint";
import { DatasetConfigNotice } from "./components/DatasetConfigNotice";
import {
  buildTrainingBlueprint,
  type BlueprintContext,
  type BlueprintStep,
  wizardSettingsValid,
} from "./trainingBlueprint";
import { WizardValidationCallout } from "./components/WizardValidationCallout";
import { wizardFirstBlockMessage, wizardStepBlockMessage } from "./wizardStepValidation";
import { MODEL_TASK_WIZARD_STEPS } from "./wizardSteps";
import { ComputeWizardStep } from "./components/ComputeWizardStep";
import type { WorkerSizePreset } from "./workerSizePresets";
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";

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
  onRequestDeleteJob: (jobId: number) => void;
  onRunJob: (jobId: number) => void;
  onStopJob: (jobId: number) => void;
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
  jobWorkerCpu: number;
  onJobWorkerCpuChange: (n: number) => void;
  jobWorkerMemoryGb: number;
  onJobWorkerMemoryGbChange: (n: number) => void;
  onCreateJob: (blueprint: ModelTaskBlueprintFile) => void | Promise<void>;
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
  onTaskTypeChange: (t: "causal_lm" | "sequence_classification") => void;
  promptFormat: PromptFormatSetting;
  onPromptFormatChange: (p: PromptFormatSetting) => void;
  intermediateSize: number;
  tieWordEmbeddings: boolean;
  baseModel?: string | null;
  useLora?: boolean;
  onApplyBlueprint: (file: ModelTaskBlueprintFile) => void;
  onProjectRaySettingsSaved?: () => void;
};

function WizardStepCard({
  title,
  description,
  children,
}: {
  title: string;
  description?: string;
  children: ReactNode;
}) {
  return (
    <article className="card model-task-wizard-step-card">
      <header className="model-task-wizard-step-header">
        <h3>{title}</h3>
        {description && <p className="hint blueprint-step-description">{description}</p>}
      </header>
      {children}
    </article>
  );
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
    onRequestDeleteJob,
    onRunJob,
    onStopJob,
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
    jobWorkerCpu,
    onJobWorkerCpuChange,
    jobWorkerMemoryGb,
    onJobWorkerMemoryGbChange,
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
    onTaskTypeChange,
    promptFormat,
    onPromptFormatChange,
    intermediateSize,
    tieWordEmbeddings,
    baseModel,
    useLora,
    onApplyBlueprint,
    onProjectRaySettingsSaved,
  } = props;

  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [workerPresetId, setWorkerPresetId] = useState<string | null>("large");

  function applyWorkerPreset(p: WorkerSizePreset) {
    setWorkerPresetId(p.id);
    onJobWorkerCpuChange(p.worker_cpu);
    onJobWorkerMemoryGbChange(p.worker_memory_gb);
    onJobGpusChange(p.gpus_per_worker);
  }

  const [savedBlueprints, setSavedBlueprints] = useState<
    { id: string; name: string; updated_at?: string }[]
  >([]);
  const [blueprintMessage, setBlueprintMessage] = useState<string | null>(null);
  const uploadRef = useRef<HTMLInputElement>(null);
  const [projectTasks, setProjectTasks] = useState<TaskDefinition[]>([]);
  const [selectedTaskId, setSelectedTaskId] = useState<number | null>(null);
  const [taskVersions, setTaskVersions] = useState<TaskVersion[]>([]);

  useEffect(() => {
    setSelectedTaskId(null);
    setTaskVersions([]);
    if (selectedProjectId == null) {
      setProjectTasks([]);
      return;
    }
    api.listProjectTasks(selectedProjectId).then(setProjectTasks).catch(() => setProjectTasks([]));
  }, [selectedProjectId]);

  useEffect(() => {
    if (selectedTaskId == null) {
      setTaskVersions([]);
      return;
    }
    api.listTaskVersions(selectedTaskId).then(setTaskVersions).catch(() => setTaskVersions([]));
  }, [selectedTaskId]);

  const reloadProjectTasks = useCallback(() => {
    if (selectedProjectId == null) return;
    api.listProjectTasks(selectedProjectId).then(setProjectTasks).catch(() => setProjectTasks([]));
  }, [selectedProjectId]);

  const wizardState: ModelTaskWizardState = useMemo(
    () => ({
      preset,
      project_id: selectedProjectId,
      job_name: jobName,
      openai_model_id: openaiModelId,
      num_workers: jobWorkers,
      gpus_per_worker: jobGpus,
      worker_cpu: jobWorkerCpu,
      worker_memory_gb: jobWorkerMemoryGb,
      max_steps: maxSteps,
      init_mode: initMode,
      task_type: taskType,
      prompt_format: promptFormat,
      arch_family: archFamily,
      vocab_size: vocabSize,
      hidden_size: hiddenSize,
      layers,
      heads,
      max_position: maxPosition,
      intermediate_size: intermediateSize,
      tie_word_embeddings: tieWordEmbeddings,
      tokenizer,
      dataset,
    }),
    [
      preset,
      selectedProjectId,
      jobName,
      openaiModelId,
      jobWorkers,
      jobGpus,
      jobWorkerCpu,
      jobWorkerMemoryGb,
      maxSteps,
      initMode,
      taskType,
      promptFormat,
      archFamily,
      vocabSize,
      hiddenSize,
      layers,
      heads,
      maxPosition,
      intermediateSize,
      tieWordEmbeddings,
      tokenizer,
      dataset,
    ],
  );

  const blueprintFile = useMemo(
    () => buildModelTaskBlueprint(wizardState, jobName || "model-task"),
    [wizardState, jobName],
  );

  useEffect(() => {
    const t = window.setTimeout(() => {
      api.putTaskBlueprintDraft(blueprintFile).catch(() => {});
    }, 700);
    return () => window.clearTimeout(t);
  }, [blueprintFile]);

  useEffect(() => {
    api.listTaskBlueprints().then(setSavedBlueprints).catch(() => setSavedBlueprints([]));
  }, []);

  const blueprintContext: BlueprintContext = useMemo(
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
      selectedProjectId,
      jobName,
      openaiModelId,
      paramEstimate,
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
      selectedProjectId,
      jobName,
      openaiModelId,
      paramEstimate,
    ],
  );

  const blueprintSteps: BlueprintStep[] = useMemo(
    () => buildTrainingBlueprint(blueprintContext),
    [blueprintContext],
  );

  const stepById = useMemo(
    () => new Map(blueprintSteps.map((s) => [s.id, s])),
    [blueprintSteps],
  );

  const currentStep = MODEL_TASK_WIZARD_STEPS[currentStepIndex];
  const currentBlueprintStep = stepById.get(currentStep.id);
  const canGoNext = currentBlueprintStep?.done ?? false;
  const stepBlock = wizardStepBlockMessage(currentStep.id, blueprintContext);
  const ready = wizardSettingsValid(blueprintContext);
  const startBlock = wizardFirstBlockMessage(blueprintContext);
  const isLastStep = currentStepIndex >= MODEL_TASK_WIZARD_STEPS.length - 1;

  const stepDescription = currentBlueprintStep?.description;

  const loadSavedBlueprint = useCallback(
    async (id: string) => {
      const raw = await api.getTaskBlueprint(id);
      const parsed = parseModelTaskBlueprintFile(raw);
      onApplyBlueprint(parsed);
      setBlueprintMessage(`Loaded blueprint “${parsed.name ?? id}”`);
    },
    [onApplyBlueprint],
  );

  const handleUpload = useCallback(
    (file: File) => {
      file
        .text()
        .then((text) => {
          const parsed = parseModelTaskBlueprintFile(JSON.parse(text));
          onApplyBlueprint(parsed);
          setBlueprintMessage(`Imported ${file.name}`);
          setCurrentStepIndex(0);
        })
        .catch((e) => {
          setBlueprintMessage(e instanceof Error ? e.message : String(e));
        });
    },
    [onApplyBlueprint],
  );

  function renderStepBody(stepId: string): ReactNode {
    switch (stepId) {
      case "project":
        return (
          <ProjectWizardStep
            projects={projects}
            selectedProjectId={selectedProjectId}
            onSelectedProjectIdChange={onSelectedProjectIdChange}
            newProjectName={newProjectName}
            onNewProjectNameChange={onNewProjectNameChange}
            newProjectDescription={newProjectDescription}
            onNewProjectDescriptionChange={onNewProjectDescriptionChange}
            onCreateProject={onCreateProject}
            projectBusy={projectBusy}
          />
        );
      case "mode":
        return (
          <FromScratchConfigFields
            section="mode"
            initMode={initMode}
            onInitModeChange={onInitModeChange}
            tokenizer={tokenizer}
            onTokenizerChange={onTokenizerChange}
            vocabSize={vocabSize}
            onVocabSizeChange={onVocabSizeChange}
            maxSteps={maxSteps}
            hiddenSize={hiddenSize}
            layers={layers}
          />
        );
      case "preset":
        return (
          <PresetWizardStep
            presets={presets}
            preset={preset}
            onPresetChange={onPresetChange}
            selectedPreset={selectedPreset}
            taskType={taskType}
            onTaskTypeChange={onTaskTypeChange}
            promptFormat={promptFormat}
            onPromptFormatChange={onPromptFormatChange}
          />
        );
      case "dataset":
        return <DatasetConfigFields value={dataset} onChange={onDatasetChange} compact />;
      case "tokenizer":
        return (
          <FromScratchConfigFields
            section="tokenizer"
            initMode={initMode}
            onInitModeChange={onInitModeChange}
            tokenizer={tokenizer}
            onTokenizerChange={onTokenizerChange}
            vocabSize={vocabSize}
            onVocabSizeChange={onVocabSizeChange}
            maxSteps={maxSteps}
            hiddenSize={hiddenSize}
            layers={layers}
          />
        );
      case "architecture":
        return (
          <ArchitectureGeometryStep
            initMode={initMode}
            presetTitle={selectedPreset?.title}
            family={archFamily}
            taskType={taskType}
            baseModel={baseModel}
            useLora={useLora}
            hiddenSize={hiddenSize}
            layers={layers}
            heads={heads}
            maxPosition={maxPosition}
            vocabSize={vocabSize}
            intermediateSize={intermediateSize}
            tieWordEmbeddings={tieWordEmbeddings}
            paramEstimate={paramEstimate}
            onHiddenSizeChange={onHiddenSizeChange}
            onLayersChange={onLayersChange}
            onHeadsChange={onHeadsChange}
            onMaxPositionChange={onMaxPositionChange}
          />
        );
      case "training":
        return (
          <div className="wizard-step-compact">
            <label className="architecture-field">
              <span className="architecture-field-label">Training steps (max_steps)</span>
              <ValidatedNumberField value={maxSteps} min={10} max={500_000} onChange={onMaxStepsChange} />
            </label>
            <p className="hint wizard-step-footnote">
              Gradient-step budget for the Ray Train loop (smoke tests often use 40–500 steps).
            </p>
          </div>
        );
      case "compute":
        return (
          <ComputeWizardStep
            selectedProjectId={selectedProjectId}
            onProjectRaySettingsSaved={onProjectRaySettingsSaved}
            workerPresetId={workerPresetId}
            startBusy={startBusy}
            jobWorkers={jobWorkers}
            jobGpus={jobGpus}
            jobWorkerCpu={jobWorkerCpu}
            jobWorkerMemoryGb={jobWorkerMemoryGb}
            onApplyWorkerPreset={applyWorkerPreset}
            onClearWorkerPreset={() => setWorkerPresetId(null)}
            onJobWorkersChange={onJobWorkersChange}
            onJobGpusChange={onJobGpusChange}
            onJobWorkerCpuChange={onJobWorkerCpuChange}
            onJobWorkerMemoryGbChange={onJobWorkerMemoryGbChange}
          />
        );
      case "register":
        return (
          <div className="wizard-step-compact architecture-form-grid architecture-form-grid-2">
            <label className="architecture-field">
              <span className="architecture-field-label">Job name</span>
              <input value={jobName} onChange={(e) => onJobNameChange(e.target.value)} />
            </label>
            <label className="architecture-field">
              <span className="architecture-field-label">OpenAI model id</span>
              <input value={openaiModelId} onChange={(e) => onOpenaiModelIdChange(e.target.value)} />
            </label>
          </div>
        );
      default:
        return null;
    }
  }

  return (
    <div className="settings-page model-tasks-page model-task-wizard">
      <header className="main-panel-header">
        <h2>Task configurations</h2>
        <p className="hint">
          Configure a versioned task JSON, then start a <strong>job run</strong>. Draft autosave:{" "}
          <code>data/blueprints/_draft.json</code>.
        </p>
      </header>

      <div className="model-task-wizard-toolbar card">
        <div className="model-task-wizard-toolbar-actions">
          <button
            type="button"
            className="btn-secondary"
            onClick={() => downloadBlueprintJson(blueprintFile)}
          >
            Download JSON
          </button>
          <button
            type="button"
            className="btn-secondary"
            onClick={() => uploadRef.current?.click()}
          >
            Upload JSON
          </button>
          <input
            ref={uploadRef}
            type="file"
            accept="application/json,.json"
            className="sr-only"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) handleUpload(f);
              e.target.value = "";
            }}
          />
          <button
            type="button"
            className="btn-secondary"
            onClick={async () => {
              if (!selectedProjectId) {
                setBlueprintMessage("Select a project first");
                return;
              }
              const name = window.prompt("Task name", jobName || "my-task");
              if (!name?.trim()) return;
              let taskId = selectedTaskId;
              if (!taskId) {
                const created = await api.createProjectTask(selectedProjectId, { name: name.trim() });
                taskId = created.id;
                setSelectedTaskId(created.id);
              }
              const ver = await api.createTaskVersion(taskId, blueprintFile as unknown as Record<string, unknown>);
              setBlueprintMessage(`Saved task version v${ver.version}`);
              reloadProjectTasks();
              setTaskVersions(await api.listTaskVersions(taskId));
            }}
          >
            Save task version
          </button>
          <button
            type="button"
            className="btn-secondary"
            onClick={async () => {
              const name = window.prompt("Blueprint file name", jobName || "my-blueprint");
              if (!name?.trim()) return;
              const file = buildModelTaskBlueprint(wizardState, name.trim());
              await api.saveTaskBlueprint(file);
              setBlueprintMessage(`Exported blueprint “${name.trim()}” to data/blueprints/`);
              const list = await api.listTaskBlueprints();
              setSavedBlueprints(list);
            }}
          >
            Export JSON file
          </button>
        </div>
        {savedBlueprints.length > 0 && (
          <label className="model-task-wizard-load">
            Load saved blueprint
            <select
              defaultValue=""
              onChange={(e) => {
                const id = e.target.value;
                if (id) void loadSavedBlueprint(id);
                e.target.value = "";
              }}
            >
              <option value="">Choose…</option>
              {savedBlueprints.map((b) => (
                <option key={b.id} value={b.id}>{b.name}</option>
              ))}
            </select>
          </label>
        )}
        {blueprintMessage && <p className="hint model-task-wizard-msg">{blueprintMessage}</p>}
      </div>

      <ModelTaskWizardStepper
        steps={[...MODEL_TASK_WIZARD_STEPS]}
        blueprintSteps={blueprintSteps}
        currentIndex={currentStepIndex}
        onSelectIndex={setCurrentStepIndex}
      />

      {currentStep.id === "dataset" && <DatasetConfigNotice dataset={dataset} />}

      <div className="model-task-wizard-body">
        <nav className="model-task-wizard-nav" aria-label="Wizard steps">
          <div className="model-task-wizard-nav-actions">
            <button
              type="button"
              className="btn-secondary"
              disabled={currentStepIndex === 0}
              onClick={() => setCurrentStepIndex((i) => Math.max(0, i - 1))}
            >
              Back
            </button>
            {!isLastStep ? (
              <button
                type="button"
                disabled={!canGoNext}
                aria-describedby={!canGoNext && stepBlock ? "wizard-next-block-reason" : undefined}
                onClick={() =>
                  setCurrentStepIndex((i) => Math.min(MODEL_TASK_WIZARD_STEPS.length - 1, i + 1))
                }
              >
                Next
              </button>
            ) : (
              <button
                type="button"
                disabled={!ready || startBusy}
                aria-describedby={!ready && startBlock ? "wizard-start-block-reason" : undefined}
                onClick={() => void onCreateJob(blueprintFile)}
              >
                {startBusy ? "Starting…" : "Start job"}
              </button>
            )}
          </div>
          {!isLastStep && !canGoNext && stepBlock && (
            <WizardValidationCallout
              id="wizard-next-block-reason"
              variant="footer"
              title="Next is disabled"
              block={stepBlock}
            />
          )}
          {isLastStep && !ready && startBlock && (
            <WizardValidationCallout
              id="wizard-start-block-reason"
              variant="footer"
              title="Start job is disabled"
              block={startBlock}
            />
          )}
        </nav>

        <WizardStepCard
          title={currentStep.title}
          description={stepDescription}
        >
          {stepBlock && <WizardValidationCallout block={stepBlock} />}
          {renderStepBody(currentStep.id)}
        </WizardStepCard>
      </div>
    </div>
  );
}
