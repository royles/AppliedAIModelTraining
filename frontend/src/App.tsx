import { useCallback, useEffect, useMemo, useState } from "react";
import {
  api,
  type Experiment,
  type TransformerPreset,
  type TrainingDatasetConfig,
  type TrainingJob,
} from "./api";
import { ModelTasksPanel } from "./ModelTasksPanel";
import { PlatformSettingsPanel } from "./PlatformSettingsPanel";
import { TaskCardGrid } from "./TaskCardGrid";
import { TaskDetailView } from "./TaskDetailView";

type NavView = "tasks" | "framework" | "model-tasks";

export default function App() {
  const [nav, setNav] = useState<NavView>("tasks");
  const [selectedJobId, setSelectedJobId] = useState<number | null>(null);

  const [jobWorkers, setJobWorkers] = useState(1);
  const [jobGpus, setJobGpus] = useState(0);
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [jobs, setJobs] = useState<TrainingJob[]>([]);
  const [presets, setPresets] = useState<TransformerPreset[]>([]);
  const [preset, setPreset] = useState("scratch_gpt2_tiny");
  const [paramEstimate, setParamEstimate] = useState<number | null>(null);
  const [expName, setExpName] = useState("Scratch transformer");
  const [jobName, setJobName] = useState("From-scratch GPT-2");
  const [openaiModelId, setOpenaiModelId] = useState("my-scratch-chat");
  const [selectedExp, setSelectedExp] = useState<number | null>(null);
  const [hiddenSize, setHiddenSize] = useState(256);
  const [layers, setLayers] = useState(4);
  const [heads, setHeads] = useState(4);
  const [maxSteps, setMaxSteps] = useState(40);
  const [dataset, setDataset] = useState<TrainingDatasetConfig>({
    source: "builtin",
    builtin_name: "platform_demo",
    repeat: 40,
  });
  const [error, setError] = useState<string | null>(null);

  const selectedPreset = presets.find((p) => p.name === preset);
  const selectedJob = useMemo(
    () => jobs.find((j) => j.id === selectedJobId) ?? null,
    [jobs, selectedJobId],
  );

  const refresh = useCallback(async () => {
    try {
      const [exps, j, pr] = await Promise.all([
        api.listExperiments(),
        api.listJobs(),
        api.listPresets(),
      ]);
      setExperiments(exps);
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
  }, []);

  useEffect(() => {
    refresh();
    const id = window.setInterval(refresh, 4000);
    return () => window.clearInterval(id);
  }, [refresh]);

  useEffect(() => {
    const arch = selectedPreset?.config.architecture;
    if (!arch) return;
    setHiddenSize(arch.hidden_size);
    setLayers(arch.num_hidden_layers);
    setHeads(arch.num_attention_heads);
    if (selectedPreset.config.dataset) {
      setDataset(selectedPreset.config.dataset);
    }
    api
      .estimateArchitecture({
        ...arch,
        hidden_size: hiddenSize,
        num_hidden_layers: layers,
        num_attention_heads: heads,
      })
      .then((r) => setParamEstimate(r.estimated_parameters))
      .catch(() => setParamEstimate(null));
  }, [selectedPreset, hiddenSize, layers, heads]);

  async function onCreateExperiment() {
    await api.createExperiment({ name: expName });
    setExpName("");
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
        init_mode: "from_scratch",
        task_type: "causal_lm",
        architecture: {
          family: "gpt2",
          vocab_size: 50257,
          hidden_size: hiddenSize,
          num_hidden_layers: layers,
          num_attention_heads: heads,
          max_position_embeddings: 512,
          tie_word_embeddings: true,
        },
        max_seq_length: 256,
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
          />
        ) : nav === "tasks" ? (
          <>
            <header className="main-panel-header">
              <h2>Model task runs</h2>
              <p className="hint">Click a card to open live framework and training logs.</p>
            </header>
            <TaskCardGrid jobs={jobs} onSelect={openTask} onDelete={deleteTask} />
          </>
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
            onHiddenSizeChange={setHiddenSize}
            onLayersChange={setLayers}
            onHeadsChange={setHeads}
            onMaxStepsChange={setMaxSteps}
            experiments={experiments}
            selectedExp={selectedExp}
            onSelectedExpChange={setSelectedExp}
            expName={expName}
            onExpNameChange={setExpName}
            onCreateExperiment={onCreateExperiment}
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
          />
        )}
      </main>
    </div>
  );
}
