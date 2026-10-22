import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  api,
  type Project,
  type RegistryModel,
  type TransformerArchitecture,
  type TransformerPreset,
  type TrainingDatasetConfig,
  type TokenizerBuildConfig,
  type PromptFormatSetting,
  type TrainingJob,
} from "./api";
import { AppSidebar, type NavView } from "./components/AppSidebar";
import { PlatformStatusBar } from "./PlatformStatusBar";
import { useSidebarCollapsed } from "./hooks/useSidebarCollapsed";
import { ModelRegistryPanel } from "./ModelRegistryPanel";
import { ModelTasksPanel } from "./ModelTasksPanel";
import { PlatformSettingsPanel } from "./PlatformSettingsPanel";
import { TaskCardGrid } from "./TaskCardGrid";
import { TaskDetailView } from "./TaskDetailView";
import { ConfirmDialog } from "./components/ConfirmDialog";
import { JobRestartDialog } from "./components/JobRestartDialog";
import { defaultResumeForRun, isJobActive } from "./jobActions";
import { trainingJobListFieldsEqual } from "./jobUtils";
import {
  DEFAULT_DATASET,
  DEFAULT_HEADS,
  DEFAULT_HIDDEN_SIZE,
  DEFAULT_JOB_NAME,
  DEFAULT_LAYERS,
  DEFAULT_MAX_STEPS,
  DEFAULT_OPENAI_MODEL_ID,
  DEFAULT_PRESET,
  DEFAULT_PROJECT_NAME,
  DEFAULT_TOKENIZER,
  DEFAULT_VOCAB_SIZE,
} from "./modelTaskDefaults";
import {
  parseModelTaskBlueprintFile,
  wizardStateFromBlueprint,
  type ModelTaskBlueprintFile,
} from "./modelTaskBlueprint";

export default function App() {
  const [sidebarCollapsed, toggleSidebar] = useSidebarCollapsed();
  const [nav, setNav] = useState<NavView>("tasks");
  const [selectedJobId, setSelectedJobId] = useState<number | null>(null);

  const [jobWorkers, setJobWorkers] = useState(1);
  const [jobGpus, setJobGpus] = useState(1);
  const [jobWorkerCpu, setJobWorkerCpu] = useState(16);
  const [jobWorkerMemoryGb, setJobWorkerMemoryGb] = useState(64);
  const [projects, setProjects] = useState<Project[]>([]);
  const [jobs, setJobs] = useState<TrainingJob[]>([]);
  const [presets, setPresets] = useState<TransformerPreset[]>([]);
  const [preset, setPreset] = useState(DEFAULT_PRESET);
  const [paramEstimate, setParamEstimate] = useState<number | null>(null);
  const [newProjectName, setNewProjectName] = useState(DEFAULT_PROJECT_NAME);
  const [newProjectDescription, setNewProjectDescription] = useState("");
  const [projectBusy, setProjectBusy] = useState(false);
  const [jobBusy, setJobBusy] = useState(false);
  const [deleteJobTarget, setDeleteJobTarget] = useState<TrainingJob | null>(null);
  const [restartJobTarget, setRestartJobTarget] = useState<TrainingJob | null>(null);
  const [startBusy, setStartBusy] = useState(false);
  const [jobName, setJobName] = useState(DEFAULT_JOB_NAME);
  const [openaiModelId, setOpenaiModelId] = useState(DEFAULT_OPENAI_MODEL_ID);
  const [selectedExp, setSelectedExp] = useState<number | null>(null);
  const [initMode, setInitMode] = useState<"from_scratch" | "pretrained">("from_scratch");
  const [tokenizer, setTokenizer] = useState<TokenizerBuildConfig>(DEFAULT_TOKENIZER);
  const [vocabSize, setVocabSize] = useState(DEFAULT_VOCAB_SIZE);
  const [hiddenSize, setHiddenSize] = useState(DEFAULT_HIDDEN_SIZE);
  const [layers, setLayers] = useState(DEFAULT_LAYERS);
  const [heads, setHeads] = useState(DEFAULT_HEADS);
  const [maxPosition, setMaxPosition] = useState(512);
  const [archFamily, setArchFamily] = useState<TransformerArchitecture["family"]>("gpt2");
  const [taskType, setTaskType] = useState<"causal_lm" | "sequence_classification">("causal_lm");
  const [promptFormat, setPromptFormat] = useState<PromptFormatSetting>("auto");
  const [intermediateSize, setIntermediateSize] = useState(DEFAULT_HIDDEN_SIZE * 4);
  const [tieWordEmbeddings, setTieWordEmbeddings] = useState(true);
  const [maxSteps, setMaxSteps] = useState(DEFAULT_MAX_STEPS);
  const [dataset, setDataset] = useState<TrainingDatasetConfig>(DEFAULT_DATASET);
  const [error, setError] = useState<string | null>(null);
  const [platformVersion, setPlatformVersion] = useState<string | null>(null);
  const [uiBundleId, setUiBundleId] = useState<string | null>(null);
  const [registryModels, setRegistryModels] = useState<RegistryModel[]>([]);
  const [registryUiMissing, setRegistryUiMissing] = useState(false);
  const [registryChatUiMissing, setRegistryChatUiMissing] = useState(false);

  const selectedPreset = presets.find((p) => p.name === preset);
  const selectedJob = useMemo(
    () => jobs.find((j) => j.id === selectedJobId) ?? null,
    [jobs, selectedJobId],
  );

  const refresh = useCallback(async () => {
    try {
      const [exps, j, pr] = await Promise.all([
        api.listProjects(),
        api.listJobs(),
        api.listPresets(),
      ]);
      setProjects(exps);
      setJobs(j);
      setPresets(pr);
      api.listRegistryModels().then(setRegistryModels).catch(() => setRegistryModels([]));
      if (selectedExp === null && exps.length > 0) {
        setSelectedExp(exps[0].id);
      }
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [selectedExp]);

  const applyProjectRayDefaults = useCallback((projectId: number) => {
    api.getProjectRaySettings(projectId).then((s) => {
      setJobWorkers(s.cdsw_workers.default_job_num_workers);
      const gpus = s.cdsw_workers.gpus_per_worker;
      setJobGpus(gpus > 0 ? gpus : 1);
      setJobWorkerCpu(s.cdsw_workers.worker_cpu);
      setJobWorkerMemoryGb(s.cdsw_workers.worker_memory_gb);
    });
  }, []);

  useEffect(() => {
    if (selectedExp != null) {
      applyProjectRayDefaults(selectedExp);
    }
  }, [selectedExp, applyProjectRayDefaults]);

  useEffect(() => {
    api.uiBuildInfo().then((info) => {
      if (info.status === "ready") {
        setPlatformVersion(info.platform_version ?? null);
        if (info.js) {
          setUiBundleId(info.js.replace(/^index-/, "").replace(/\.js$/, ""));
        }
        const features = info.features as {
          model_registry?: boolean;
          registry_chat?: boolean;
        } | undefined;
        setRegistryUiMissing(features?.model_registry === false);
        setRegistryChatUiMissing(
          features?.model_registry === true && features?.registry_chat === false,
        );
      }
    }).catch(() => {
      setPlatformVersion(null);
      setUiBundleId(null);
    });
    api.listRegistryModels().then(setRegistryModels).catch(() => setRegistryModels([]));
  }, []);

  useEffect(() => {
    refresh();
    const id = window.setInterval(refresh, 4000);
    return () => window.clearInterval(id);
  }, [refresh]);

  const appliedPresetRef = useRef<string | null>(null);
  const draftLoadedRef = useRef(false);

  const applyModelTaskBlueprint = useCallback((b: ModelTaskBlueprintFile) => {
    const s = wizardStateFromBlueprint(b);
    appliedPresetRef.current = s.preset;
    setPreset(s.preset);
    if (s.project_id != null) setSelectedExp(s.project_id);
    setJobName(s.job_name);
    setOpenaiModelId(s.openai_model_id);
    setJobWorkers(s.num_workers);
    setJobGpus(s.gpus_per_worker);
    if (s.worker_cpu != null) setJobWorkerCpu(s.worker_cpu);
    if (s.worker_memory_gb != null) setJobWorkerMemoryGb(s.worker_memory_gb);
    setMaxSteps(s.max_steps);
    setInitMode(s.init_mode);
    setTaskType(s.task_type);
    setPromptFormat(s.prompt_format ?? "auto");
    setArchFamily(s.arch_family);
    setVocabSize(s.vocab_size);
    setHiddenSize(s.hidden_size);
    setLayers(s.layers);
    setHeads(s.heads);
    setMaxPosition(s.max_position);
    setIntermediateSize(s.intermediate_size);
    setTieWordEmbeddings(s.tie_word_embeddings);
    setTokenizer(s.tokenizer);
    setDataset(s.dataset);
  }, []);

  useEffect(() => {
    if (nav !== "model-tasks" || draftLoadedRef.current) return;
    draftLoadedRef.current = true;
    api
      .getTaskBlueprintDraft()
      .then((r) => {
        if (r.blueprint) {
          applyModelTaskBlueprint(parseModelTaskBlueprintFile(r.blueprint));
        }
      })
      .catch(() => {});
  }, [nav, applyModelTaskBlueprint]);

  useEffect(() => {
    const p = presets.find((x) => x.name === preset);
    if (!p) return;
    // Presets list refreshes on a timer; only re-apply when the user picks a different preset.
    if (appliedPresetRef.current === preset) return;
    appliedPresetRef.current = preset;

    const arch = p.config.architecture;
    const c = p.config;
    setHiddenSize(arch.hidden_size);
    setLayers(arch.num_hidden_layers);
    setHeads(arch.num_attention_heads);
    setMaxPosition(arch.max_position_embeddings);
    setArchFamily(arch.family);
    setTaskType(c.task_type);
    setIntermediateSize(arch.intermediate_size ?? arch.hidden_size * 4);
    setTieWordEmbeddings(arch.tie_word_embeddings);
    if (c.max_steps) setMaxSteps(c.max_steps);
    if (c.dataset) setDataset(c.dataset);
    if (c.tokenizer) {
      setTokenizer(c.tokenizer);
      if (c.tokenizer.vocab_size) setVocabSize(c.tokenizer.vocab_size);
    }
    setInitMode(c.init_mode ?? "from_scratch");
  }, [preset, presets]);

  useEffect(() => {
    if (initMode !== "from_scratch") {
      setParamEstimate(null);
      return;
    }
    api
      .estimateArchitecture({
        family: archFamily,
        vocab_size: vocabSize,
        hidden_size: hiddenSize,
        num_hidden_layers: layers,
        num_attention_heads: heads,
        intermediate_size: intermediateSize,
        max_position_embeddings: maxPosition,
        tie_word_embeddings: tieWordEmbeddings,
      })
      .then((r) => setParamEstimate(r.estimated_parameters))
      .catch(() => setParamEstimate(null));
  }, [
    initMode,
    archFamily,
    vocabSize,
    hiddenSize,
    layers,
    heads,
    maxPosition,
    intermediateSize,
    tieWordEmbeddings,
  ]);

  function handleHiddenSizeChange(n: number) {
    setHiddenSize(n);
    setIntermediateSize(n * 4);
  }

  function handleVocabSizeChange(n: number) {
    setVocabSize(n);
    setTokenizer((t) => ({ ...t, vocab_size: n }));
  }

  function handleTaskTypeChange(next: "causal_lm" | "sequence_classification") {
    setTaskType(next);
    setArchFamily(next === "causal_lm" ? "gpt2" : "bert");
  }

  const handleJobUpdated = useCallback((fresh: TrainingJob) => {
    setJobs((prev) => {
      const existing = prev.find((j) => j.id === fresh.id);
      if (existing && trainingJobListFieldsEqual(existing, fresh)) {
        return prev;
      }
      return prev.map((j) => (j.id === fresh.id ? fresh : j));
    });
  }, []);

  async function onCreateProject() {
    const name = newProjectName.trim();
    if (!name) return;
    setProjectBusy(true);
    try {
      const created = await api.createProject({
        name,
        description: newProjectDescription.trim() || undefined,
      });
      setNewProjectName("");
      setNewProjectDescription("");
      setSelectedExp(created.id);
      await refresh();
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setProjectBusy(false);
    }
  }

  async function onUpdateProject(projectId: number, name: string) {
    setProjectBusy(true);
    try {
      await api.updateProject(projectId, { name });
      await refresh();
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setProjectBusy(false);
    }
  }

  async function onDeleteProject(projectId: number) {
    if (!window.confirm("Delete this project and all tasks in it?")) return;
    setProjectBusy(true);
    try {
      await api.deleteProject(projectId);
      if (selectedExp === projectId) {
        setSelectedExp(null);
      }
      await refresh();
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setProjectBusy(false);
    }
  }

  async function onUpdateJobName(jobId: number, name: string) {
    setJobBusy(true);
    try {
      await api.updateJob(jobId, { name });
      await refresh();
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setJobBusy(false);
    }
  }

  function requestDeleteJob(jobId: number) {
    const job = jobs.find((j) => j.id === jobId) ?? null;
    if (job) {
      setDeleteJobTarget(job);
    }
  }

  async function confirmDeleteJob() {
    if (!deleteJobTarget) return;
    setJobBusy(true);
    try {
      if (isJobActive(deleteJobTarget.status)) {
        try {
          await api.stopJob(deleteJobTarget.id);
        } catch {
          /* proceed with delete if stop races with completion */
        }
      }
      await api.deleteJob(deleteJobTarget.id);
      if (selectedJobId === deleteJobTarget.id) {
        closeTask();
        setNav("tasks");
      }
      setDeleteJobTarget(null);
      await refresh();
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setJobBusy(false);
    }
  }

  async function onCreateJob(blueprint: ModelTaskBlueprintFile) {
    if (!selectedExp) return;
    setStartBusy(true);
    try {
    await api.createJob({
      experiment_id: selectedExp,
      name: jobName,
      preset,
      openai_model_id: openaiModelId,
      num_workers: jobWorkers,
      gpus_per_worker: jobGpus,
      worker_cpu: jobWorkerCpu,
      worker_memory_gb: jobWorkerMemoryGb,
      max_steps: maxSteps,
      task_blueprint: blueprint,
      save_as_task: true,
      task_name: jobName,
      transformer: {
        init_mode: initMode,
        task_type: taskType,
        prompt_format: taskType === "causal_lm" ? promptFormat : "auto",
        tokenizer,
        architecture: {
          family: archFamily,
          vocab_size: vocabSize,
          hidden_size: hiddenSize,
          num_hidden_layers: layers,
          num_attention_heads: heads,
          intermediate_size: intermediateSize,
          max_position_embeddings: maxPosition,
          tie_word_embeddings: tieWordEmbeddings,
        },
        max_seq_length: Math.min(256, maxPosition),
        per_device_train_batch_size: 2,
        max_steps: maxSteps,
        learning_rate: 6e-4,
        dataset,
      },
    });
    await refresh();
    setNav("tasks");
    setSelectedJobId(null);
    setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setStartBusy(false);
    }
  }

  function openTask(jobId: number) {
    setSelectedJobId(jobId);
  }

  function closeTask() {
    setSelectedJobId(null);
  }

  function goNav(view: NavView) {
    setNav(view);
    setSelectedJobId(null);
  }

  function openRegistry() {
    setSelectedJobId(null);
    setNav("registry");
  }

  async function restartTask(jobId: number, resume: boolean) {
    setJobBusy(true);
    try {
      const updated = await api.restartJob(jobId, { resume });
      setJobs((prev) => prev.map((j) => (j.id === jobId ? updated : j)));
      setRestartJobTarget(null);
      setError(null);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setJobBusy(false);
    }
  }

  function handleRunJob(jobId: number) {
    const job = jobs.find((j) => j.id === jobId);
    if (!job) return;
    if (job.status === "failed" || job.status === "stopped") {
      setRestartJobTarget(job);
      return;
    }
    void restartTask(jobId, defaultResumeForRun(job.status));
  }

  async function stopJob(jobId: number) {
    setJobBusy(true);
    try {
      const updated = await api.stopJob(jobId);
      setJobs((prev) => prev.map((j) => (j.id === jobId ? updated : j)));
      setError(null);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setJobBusy(false);
    }
  }

  return (
    <div className="app-root">
      <a href="#main-content" className="skip-link">
        Skip to main content
      </a>
      <PlatformStatusBar jobs={jobs} onOpenTask={openTask} />
      <div className="app-shell">
        <AppSidebar
          collapsed={sidebarCollapsed}
          onToggleCollapse={toggleSidebar}
          nav={nav}
          selectedJobId={selectedJobId}
          jobCount={jobs.length}
          registryCount={registryModels.length}
          platformVersion={platformVersion}
          uiBundleId={uiBundleId}
          onNavigate={goNav}
        />

      <main id="main-content" className="main-panel" tabIndex={-1}>
        <header className="main-panel-header compact">
          <p className="subtitle">
            Ray · CDSW workers · OpenAI-compatible <code>/v1</code> API
          </p>
        </header>

        {error && <div className="banner error">{error}</div>}
        {registryUiMissing && (
          <div className="banner warn">
            This UI bundle is missing the Model registry panel. Run{" "}
            <code>git pull origin main</code> and restart <code>scripts/start.py</code>, then hard-refresh
            the browser.
          </div>
        )}
        {registryChatUiMissing && (
          <div className="banner warn">
            This UI bundle is outdated (registry chat missing). Run <code>git pull origin main</code>,
            restart <code>python3 scripts/start.py</code>, and hard-refresh (Ctrl+Shift+R).
          </div>
        )}

        {selectedJob && selectedJobId !== null ? (
          <TaskDetailView
            job={selectedJob}
            onBack={() => {
              closeTask();
              setNav("tasks");
            }}
            onRequestDelete={() => setDeleteJobTarget(selectedJob)}
            onRun={() => handleRunJob(selectedJob.id)}
            onStop={() => stopJob(selectedJob.id)}
            actionBusy={jobBusy}
            onJobUpdated={handleJobUpdated}
            onOpenRegistry={openRegistry}
          />
        ) : nav === "tasks" ? (
          <>
            <header className="main-panel-header">
              <h2>Job runs</h2>
              <p className="hint">
                Executions of task configurations. Click a card for live logs. Trained models appear in{" "}
                <button type="button" className="link-button" onClick={openRegistry}>
                  Model registry
                </button>
                .
              </p>
            </header>
            <TaskCardGrid
              projects={projects}
              jobs={jobs}
              onSelect={openTask}
              onRequestDelete={requestDeleteJob}
              onRun={handleRunJob}
              onStop={stopJob}
              jobBusy={jobBusy}
            />
          </>
        ) : nav === "registry" ? (
          <ModelRegistryPanel />
        ) : nav === "framework" ? (
          <div className="settings-page">
            <header className="main-panel-header">
              <h2>Framework Configuration</h2>
              <p className="hint">
                Platform defaults for <strong>new projects</strong>. Each project keeps its own Ray
                configuration (Model Tasks → Project step).
              </p>
            </header>
            <PlatformSettingsPanel title="Platform defaults (new projects)" />
          </div>
        ) : (
          <ModelTasksPanel
            presets={presets}
            preset={preset}
            onPresetChange={setPreset}
            selectedPreset={selectedPreset}
            hiddenSize={hiddenSize}
            layers={layers}
            heads={heads}
            maxSteps={maxSteps}
            paramEstimate={paramEstimate}
            onHiddenSizeChange={handleHiddenSizeChange}
            onLayersChange={setLayers}
            onHeadsChange={setHeads}
            onMaxStepsChange={setMaxSteps}
            projects={projects}
            jobs={jobs}
            selectedProjectId={selectedExp}
            onSelectedProjectIdChange={setSelectedExp}
            newProjectName={newProjectName}
            onNewProjectNameChange={setNewProjectName}
            newProjectDescription={newProjectDescription}
            onNewProjectDescriptionChange={setNewProjectDescription}
            onCreateProject={onCreateProject}
            onUpdateProject={onUpdateProject}
            onDeleteProject={onDeleteProject}
            onUpdateJobName={onUpdateJobName}
            onRequestDeleteJob={requestDeleteJob}
            onRunJob={handleRunJob}
            onStopJob={stopJob}
            projectBusy={projectBusy}
            jobBusy={jobBusy}
            startBusy={startBusy}
            jobName={jobName}
            onJobNameChange={setJobName}
            openaiModelId={openaiModelId}
            onOpenaiModelIdChange={setOpenaiModelId}
            jobWorkers={jobWorkers}
            onJobWorkersChange={setJobWorkers}
            jobGpus={jobGpus}
            onJobGpusChange={setJobGpus}
            jobWorkerCpu={jobWorkerCpu}
            onJobWorkerCpuChange={setJobWorkerCpu}
            jobWorkerMemoryGb={jobWorkerMemoryGb}
            onJobWorkerMemoryGbChange={setJobWorkerMemoryGb}
            onCreateJob={onCreateJob}
            onApplyBlueprint={applyModelTaskBlueprint}
            onProjectRaySettingsSaved={() => {
              if (selectedExp != null) applyProjectRayDefaults(selectedExp);
            }}
            dataset={dataset}
            onDatasetChange={setDataset}
            initMode={initMode}
            onInitModeChange={setInitMode}
            tokenizer={tokenizer}
            onTokenizerChange={setTokenizer}
            vocabSize={vocabSize}
            onVocabSizeChange={handleVocabSizeChange}
            maxPosition={maxPosition}
            onMaxPositionChange={setMaxPosition}
            archFamily={archFamily}
            taskType={taskType}
            onTaskTypeChange={handleTaskTypeChange}
            promptFormat={promptFormat}
            onPromptFormatChange={setPromptFormat}
            intermediateSize={intermediateSize}
            tieWordEmbeddings={tieWordEmbeddings}
            baseModel={selectedPreset?.config.base_model}
            useLora={selectedPreset?.config.use_lora ?? false}
          />
        )}
      </main>
      </div>

      <ConfirmDialog
        open={deleteJobTarget !== null}
        title="Delete job?"
        confirmLabel="Delete"
        danger
        busy={jobBusy}
        onCancel={() => setDeleteJobTarget(null)}
        onConfirm={() => void confirmDeleteJob()}
      >
        {deleteJobTarget ? (
          <>
            <p>
              Delete run <strong>#{deleteJobTarget.id}</strong> &ldquo;{deleteJobTarget.name}&rdquo;?
            </p>
            <p className="hint">
              This removes the job record, logs, and any registered <code>/v1</code> model entry tied to this
              run. Artifacts on disk are removed when possible. Active workers may take a moment to shut down.
            </p>
          </>
        ) : null}
      </ConfirmDialog>

      <JobRestartDialog
        open={restartJobTarget !== null}
        jobName={restartJobTarget?.name ?? ""}
        busy={jobBusy}
        onCancel={() => setRestartJobTarget(null)}
        onResume={() => restartJobTarget && void restartTask(restartJobTarget.id, true)}
        onStartOver={() => restartJobTarget && void restartTask(restartJobTarget.id, false)}
      />
    </div>
  );
}
