import { useCallback, useEffect, useState } from "react";
import {
  api,
  type Experiment,
  type TransformerPreset,
  type TrainingJob,
} from "./api";
import { JobRunList } from "./JobRunList";
import { PlatformSettingsPanel } from "./PlatformSettingsPanel";
import { TransformerArchitectureDiagram } from "./TransformerArchitectureDiagram";

export default function App() {
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
  const [error, setError] = useState<string | null>(null);

  const selectedPreset = presets.find((p) => p.name === preset);

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
      },
    });
    await refresh();
  }

  return (
    <div className="layout">
      <header>
        <h1>Applied AI Model Training</h1>
        <p className="subtitle">
          Build transformers from scratch · Ray · OpenAI-compatible <code>/v1</code> API
        </p>
      </header>

      {error && <div className="banner error">{error}</div>}

      <section className="grid">
        <PlatformSettingsPanel
          onSaved={() => {
            api.getPlatformSettings().then((s) => {
              setJobWorkers(s.cdsw_workers.default_job_num_workers);
              setJobGpus(s.cdsw_workers.gpus_per_worker);
            });
          }}
        />

        <article className="card">
          <h2>Experiment</h2>
          <label>
            Name
            <input value={expName} onChange={(e) => setExpName(e.target.value)} />
          </label>
          <button type="button" onClick={onCreateExperiment}>Create experiment</button>
        </article>

        <article className="card wide">
          <h2>Transformer architecture (from scratch)</h2>
          <label>
            Preset
            <select value={preset} onChange={(e) => setPreset(e.target.value)}>
              {presets.map((p) => (
                <option key={p.name} value={p.name}>{p.title}</option>
              ))}
            </select>
          </label>
          <TransformerArchitectureDiagram
            hiddenSize={hiddenSize}
            layers={layers}
            heads={heads}
            vocabSize={selectedPreset?.config.architecture.vocab_size ?? 50257}
            maxPosition={
              selectedPreset?.config.architecture.max_position_embeddings ?? 512
            }
            paramEstimate={paramEstimate}
            onHiddenSizeChange={setHiddenSize}
            onLayersChange={setLayers}
            onHeadsChange={setHeads}
          />
          <div className="arch-grid arch-grid-compact">
            <label>
              Hidden size
              <input
                type="number"
                value={hiddenSize}
                onChange={(e) => setHiddenSize(Number(e.target.value))}
              />
            </label>
            <label>
              Layers
              <input
                type="number"
                value={layers}
                onChange={(e) => setLayers(Number(e.target.value))}
              />
            </label>
            <label>
              Attention heads
              <input
                type="number"
                value={heads}
                onChange={(e) => setHeads(Number(e.target.value))}
              />
            </label>
            <label>
              Training steps
              <input
                type="number"
                value={maxSteps}
                onChange={(e) => setMaxSteps(Number(e.target.value))}
              />
            </label>
          </div>
          <label>
            Experiment
            <select
              value={selectedExp ?? ""}
              onChange={(e) => setSelectedExp(Number(e.target.value))}
            >
              {experiments.map((e) => (
                <option key={e.id} value={e.id}>{e.name}</option>
              ))}
            </select>
          </label>
          <label>
            OpenAI model id
            <input value={openaiModelId} onChange={(e) => setOpenaiModelId(e.target.value)} />
          </label>
          <label>
            Job name
            <input value={jobName} onChange={(e) => setJobName(e.target.value)} />
          </label>
          <button type="button" onClick={onCreateJob} disabled={!selectedExp}>
            Train &amp; register for /v1/chat/completions
          </button>
        </article>
      </section>

      <section className="card jobs-section">
        <h2>Training runs</h2>
        <JobRunList jobs={jobs} />
      </section>
    </div>
  );
}
