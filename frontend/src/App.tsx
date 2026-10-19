import { useCallback, useEffect, useMemo, useState } from "react";
import {
  api,
  type Project,
  type TransformerArchitecture,
  type TransformerPreset,
  type TrainingDatasetConfig,
  type TokenizerBuildConfig,
  type TrainingJob,
} from "./api";
import { ModelRegistryPanel } from "./ModelRegistryPanel";
import { ModelTasksPanel } from "./ModelTasksPanel";
import { PlatformSettingsPanel } from "./PlatformSettingsPanel";
import { TaskCardGrid } from "./TaskCardGrid";
import { TaskDetailView } from "./TaskDetailView";
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

type NavView = "tasks" | "registry" | "framework" | "model-tasks";

export default function App() {
  const [nav, setNav] = useState<NavView>("tasks");
  const [selectedJobId, setSelectedJobId] = useState<number | null>(null);

  const [jobWorkers, setJobWorkers] = useState(1);
  const [jobGpus, setJobGpus] = useState(0);
  const [projects, setProjects] = useState<Project[]>([]);
  const [jobs, setJobs] = useState<TrainingJob[]>([]);
  const [presets, setPresets] = useState<TransformerPreset[]>([]);
  const [preset, setPreset] = useState(DEFAULT_PRESET);
  const [paramEstimate, setParamEstimate] = useState<number | null>(null);
  const [projectName, setProjectName] = useState(DEFAULT_PROJECT_NAME);
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
  const [intermediateSize, setIntermediateSize] = useState(DEFAULT_HIDDEN_SIZE * 4);
  const [tieWordEmbeddings, setTieWordEmbeddings] = useState(true);
  const [maxSteps, setMaxSteps] = useState(DEFAULT_MAX_STEPS);
  const [dataset, setDataset] = useState<TrainingDatasetConfig>(DEFAULT_DATASET);
  const [error, setError] = useState<string | null>(null);
  const [uiBuildLabel, setUiBuildLabel] = useState<string | null>(null);

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
      if (selectedExp === null && exps.length > 0) {
        setSelectedExp(exps[0].id);
      }
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [selectedExp]);

  useEffect(() => {
    api.getPlatformSettings().then((s) => {
      setJobWorkers(s.cdsw_workers.default_job_num_workers);
      setJobGpus(s.cdsw_workers.gpus_per_worker);
    });
    api.uiBuildInfo().then((info) => {
      if (info.status === "ready" && info.js) {
        setUiBuildLabel(info.js.replace(/^index-/, "").replace(/\.js$/, ""));
      }
    }).catch(() => setUiBuildLabel(null));
  }, []);

  useEffect(() => {
    refresh();
    const id = window.setInterval(refresh, 4000);
    return () => window.clearInterval(id);
  }, [refresh]);

  useEffect(() => {
    const p = presets.find((x) => x.name === preset);
    if (!p) return;
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

  async function onCreateProject() {
    await api.createProject({ name: projectName });
    setProjectName("");
    await refresh();
  }

  async function onCreateJob() {
    if (!selectedExp) return;
    await api.createJob({
      experiment_id: selectedExp,
      name: jobName,
      preset,
      openai_model_id: openaiModelId,
      num_workers: jobWorkers,
      gpus_per_worker: jobGpus,
      max_steps: maxSteps,
      transformer: {
        init_mode: initMode,
        task_type: taskType,
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

  async function restartTask(jobId: number, resume: boolean) {
    try {
      const updated = await api.restartJob(jobId, { resume });
      setJobs((prev) => prev.map((j) => (j.id === jobId ? updated : j)));
      setError(null);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  async function deleteTask(jobId: number) {
    try {
      await api.deleteJob(jobId);
      if (selectedJobId === jobId) {
        closeTask();
        setNav("tasks");
      }
      await refresh();
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="sidebar-brand">
          <h1>Applied AI</h1>
          <p className="subtitle">Model Training</p>
        </div>
        <nav className="sidebar-nav" aria-label="Main">
          <button
            type="button"
            className={nav === "tasks" && selectedJobId === null ? "active" : ""}
            onClick={() => goNav("tasks")}
          >
            Tasks
          </button>
          <button
            type="button"
            className={nav === "registry" ? "active" : ""}
            onClick={() => goNav("registry")}
          >
            Model registry
          </button>
          <button
            type="button"
            className={nav === "framework" ? "active" : ""}
            onClick={() => goNav("framework")}
          >
            Framework Configuration
          </button>
          <button
            type="button"
            className={nav === "model-tasks" ? "active" : ""}
            onClick={() => goNav("model-tasks")}
          >
            Model Tasks
          </button>
        </nav>
        {uiBuildLabel && (
          <p className="hint sidebar-ui-build" title="UI bundle id from /api/ui/build-info">
            UI build {uiBuildLabel.slice(0, 8)}
          </p>
        )}
      </aside>

      <main className="main-panel">
        <header className="main-panel-header compact">
          <p className="subtitle">
            Ray · CDSW workers · OpenAI-compatible <code>/v1</code> API
          </p>
        </header>

        {error && <div className="banner error">{error}</div>}

        {selectedJob && selectedJobId !== null ? (
          <TaskDetailView
            job={selectedJob}
            onBack={() => {
              closeTask();
              setNav("tasks");
            }}
            onDelete={deleteTask}
            onRestart={restartTask}
            onJobUpdated={(fresh) => {
              setJobs((prev) => prev.map((j) => (j.id === fresh.id ? fresh : j)));
            }}
          />
        ) : nav === "tasks" ? (
          <>
            <header className="main-panel-header">
              <h2>Model task runs</h2>
              <p className="hint">Click a card to open live framework and training logs.</p>
            </header>
            <TaskCardGrid jobs={jobs} onSelect={openTask} onDelete={deleteTask} />
          </>
        ) : nav === "registry" ? (
          <ModelRegistryPanel />
        ) : nav === "framework" ? (
          <div className="settings-page">
            <header className="main-panel-header">
              <h2>Framework Configuration</h2>
              <p className="hint">Ray runtime and CDSW worker defaults for the platform.</p>
            </header>
            <PlatformSettingsPanel
              title="Ray &amp; CDSW workers"
              onSaved={() => {
                api.getPlatformSettings().then((s) => {
                  setJobWorkers(s.cdsw_workers.default_job_num_workers);
                  setJobGpus(s.cdsw_workers.gpus_per_worker);
                });
              }}
            />
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
            selectedProjectId={selectedExp}
            onSelectedProjectIdChange={setSelectedExp}
            projectName={projectName}
            onProjectNameChange={setProjectName}
            onCreateProject={onCreateProject}
            jobName={jobName}
            onJobNameChange={setJobName}
            openaiModelId={openaiModelId}
            onOpenaiModelIdChange={setOpenaiModelId}
            jobWorkers={jobWorkers}
            onJobWorkersChange={setJobWorkers}
            jobGpus={jobGpus}
            onJobGpusChange={setJobGpus}
            onCreateJob={onCreateJob}
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
            intermediateSize={intermediateSize}
            tieWordEmbeddings={tieWordEmbeddings}
            baseModel={selectedPreset?.config.base_model}
            useLora={selectedPreset?.config.use_lora ?? false}
          />
        )}
      </main>
    </div>
  );
}
