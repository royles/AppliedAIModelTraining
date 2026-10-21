import type {
  Project,
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
import { ProjectJobsHierarchy } from "./ProjectJobsHierarchy";
import { TransformerArchitectureDiagram } from "./TransformerArchitectureDiagram";
import { ValidatedNumberField } from "./ValidatedNumberField";
import {
  buildModelTaskBlueprint,
  downloadBlueprintJson,
  parseModelTaskBlueprintFile,
  type ModelTaskBlueprintFile,
  type ModelTaskWizardState,
} from "./modelTaskBlueprint";
import { blueprintReady, buildTrainingBlueprint, type BlueprintStep, type DatasetPreview } from "./trainingBlueprint";
import { MODEL_TASK_WIZARD_STEPS } from "./wizardSteps";
import { WorkerSizePresetBar } from "./components/WorkerSizePresetBar";
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
    intermediateSize,
    tieWordEmbeddings,
    baseModel,
    useLora,
    onApplyBlueprint,
    onProjectRaySettingsSaved,
  } = props;

  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [preview, setPreview] = useState<DatasetPreview | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [minCorpusChars, setMinCorpusChars] = useState<number | null>(null);
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
        dataset,
      })
      .then((g) => setMinCorpusChars((g.min_corpus_characters as number) ?? null))
      .catch(() => setMinCorpusChars(null));
  }, [initMode, tokenizer, vocabSize, hiddenSize, layers, heads, maxSteps, dataset]);

  useEffect(() => {
    api
      .datasetPreview(dataset)
      .then((p) => {
        setPreview(p);
        setPreviewError(null);
      })
      .catch((e) => {
        setPreview(null);
        setPreviewError(e instanceof Error ? e.message : String(e));
      });
  }, [dataset]);

  const blueprintSteps: BlueprintStep[] = useMemo(
    () =>
      buildTrainingBlueprint({
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
      selectedProjectId,
      jobName,
      openaiModelId,
      paramEstimate,
      preview,
      minCorpusChars,
    ],
  );

  const stepById = useMemo(
    () => new Map(blueprintSteps.map((s) => [s.id, s])),
    [blueprintSteps],
  );

  const currentStep = MODEL_TASK_WIZARD_STEPS[currentStepIndex];
  const currentBlueprintStep = stepById.get(currentStep.id);
  const canGoNext = currentBlueprintStep?.done ?? false;
  const ready = blueprintReady(blueprintSteps);
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
          <>
            <ProjectJobsHierarchy
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
              onRequestDeleteJob={onRequestDeleteJob}
              onRunJob={onRunJob}
              onStopJob={onStopJob}
              projectBusy={projectBusy}
              jobBusy={jobBusy}
              onInvalidateBlueprint={() => {}}
              taskDefinitions={projectTasks}
              taskVersions={taskVersions}
              selectedTaskId={selectedTaskId}
              onSelectedTaskIdChange={setSelectedTaskId}
              onApplyTaskVersion={(spec) => {
                onApplyBlueprint(parseModelTaskBlueprintFile(spec));
                setBlueprintMessage("Loaded task version into wizard");
              }}
            />
            {selectedProjectId != null && (
              <PlatformSettingsPanel
                projectId={selectedProjectId}
                title="Project Ray configuration"
                onSaved={onProjectRaySettingsSaved}
              />
            )}
          </>
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
      case "dataset":
        return (
          <>
            {previewError && dataset.source !== "builtin" && (
              <p className="hint blueprint-warn">Dataset preview: {previewError}</p>
            )}
            {preview && (
              <p className="hint">
                Preview: {preview.examples.toLocaleString()} examples ·{" "}
                {preview.total_characters.toLocaleString()} characters
              </p>
            )}
            <DatasetConfigFields value={dataset} onChange={onDatasetChange} />
          </>
        );
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
          <>
            <label>
              Preset
              <select value={preset} onChange={(e) => onPresetChange(e.target.value)}>
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
              onHiddenSizeChange={onHiddenSizeChange}
              onLayersChange={onLayersChange}
              onHeadsChange={onHeadsChange}
            />
            <div className="arch-grid arch-grid-compact">
              <label>
                Hidden size
                <ValidatedNumberField value={hiddenSize} min={1} max={8192} onChange={onHiddenSizeChange} />
              </label>
              <label>
                Layers
                <ValidatedNumberField value={layers} min={1} max={128} onChange={onLayersChange} />
              </label>
              <label>
                Attention heads
                <ValidatedNumberField value={heads} min={1} max={128} onChange={onHeadsChange} />
              </label>
              <label>
                Max positions
                <ValidatedNumberField value={maxPosition} min={128} max={32768} onChange={onMaxPositionChange} />
              </label>
            </div>
          </>
        );
      case "training":
        return (
          <label>
            Training steps (max_steps)
            <ValidatedNumberField value={maxSteps} min={10} max={500_000} onChange={onMaxStepsChange} />
          </label>
        );
      case "compute":
        return (
          <>
            <WorkerSizePresetBar
              selectedPresetId={workerPresetId}
              disabled={startBusy}
              onApply={applyWorkerPreset}
            />
            <div className="arch-grid">
              <label>
                Ray / CDSW worker engines (nodes)
                <ValidatedNumberField value={jobWorkers} min={1} max={32} onChange={onJobWorkersChange} />
              </label>
              <label>
                CPUs per engine
                <ValidatedNumberField
                  value={jobWorkerCpu}
                  min={0.5}
                  max={128}
                  integer={false}
                  onChange={(n) => {
                    setWorkerPresetId(null);
                    onJobWorkerCpuChange(n);
                  }}
                />
              </label>
              <label>
                Memory per engine (GiB)
                <ValidatedNumberField
                  value={jobWorkerMemoryGb}
                  min={8}
                  max={512}
                  onChange={(n) => {
                    setWorkerPresetId(null);
                    onJobWorkerMemoryGbChange(n);
                  }}
                />
              </label>
              <label>
                GPUs per engine
                <ValidatedNumberField
                  value={jobGpus}
                  min={1}
                  max={8}
                  onChange={(n) => {
                    setWorkerPresetId(null);
                    onJobGpusChange(n);
                  }}
                />
              </label>
            </div>
            <p className="hint">
              Set <strong>CML resource profile / accelerator IDs</strong> under Framework Configuration so
              each engine lands on GPU-capable hardware. Turn off low-memory mode there for distributed
              GPU training.
            </p>
          </>
        );
      case "register":
        return (
          <>
            <label>
              OpenAI model id
              <input value={openaiModelId} onChange={(e) => onOpenaiModelIdChange(e.target.value)} />
            </label>
            <label>
              Job name
              <input value={jobName} onChange={(e) => onJobNameChange(e.target.value)} />
            </label>
          </>
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

      <div className="model-task-wizard-body">
        <WizardStepCard
          title={currentStep.title}
          description={stepDescription}
        >
          {currentBlueprintStep && !currentBlueprintStep.done && currentBlueprintStep.requirement && (
            <p className="hint blueprint-warn">{currentBlueprintStep.requirement}</p>
          )}
          {renderStepBody(currentStep.id)}
        </WizardStepCard>

        <footer className="model-task-wizard-footer">
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
              onClick={() => setCurrentStepIndex((i) => Math.min(MODEL_TASK_WIZARD_STEPS.length - 1, i + 1))}
            >
              Next
            </button>
          ) : (
            <button
              type="button"
              disabled={!ready || startBusy}
              onClick={() => void onCreateJob(blueprintFile)}
            >
              {startBusy ? "Starting…" : "Start job"}
            </button>
          )}
        </footer>
      </div>
    </div>
  );
}
