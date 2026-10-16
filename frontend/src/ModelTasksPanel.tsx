import type { Experiment, TransformerPreset } from "./api";
import { TransformerArchitectureDiagram } from "./TransformerArchitectureDiagram";

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
  experiments: Experiment[];
  selectedExp: number | null;
  onSelectedExpChange: (id: number) => void;
  expName: string;
  onExpNameChange: (v: string) => void;
  onCreateExperiment: () => void;
  jobName: string;
  onJobNameChange: (v: string) => void;
  openaiModelId: string;
  onOpenaiModelIdChange: (v: string) => void;
  jobWorkers: number;
  onJobWorkersChange: (n: number) => void;
  jobGpus: number;
  onJobGpusChange: (n: number) => void;
  onCreateJob: () => void;
};

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
    experiments,
    selectedExp,
    onSelectedExpChange,
    expName,
    onExpNameChange,
    onCreateExperiment,
    jobName,
    onJobNameChange,
    openaiModelId,
    onOpenaiModelIdChange,
    jobWorkers,
    onJobWorkersChange,
    jobGpus,
    onJobGpusChange,
    onCreateJob,
  } = props;

  return (
    <div className="settings-page">
      <header className="main-panel-header">
        <h2>Model Tasks</h2>
        <p className="hint">
          Defaults for each training run: architecture, hyperparameters, workers, and experiment.
        </p>
      </header>

      <article className="card">
        <h3>Experiment</h3>
        <label>
          New experiment name
          <input value={expName} onChange={(e) => onExpNameChange(e.target.value)} />
        </label>
        <button type="button" onClick={onCreateExperiment}>Create experiment</button>
      </article>

      <article className="card">
        <h3>Per-run worker settings</h3>
        <div className="arch-grid">
          <label>
            Ray / CDSW workers for new jobs
            <input
              type="number"
              min={1}
              max={32}
              value={jobWorkers}
              onChange={(e) => onJobWorkersChange(Number(e.target.value))}
            />
          </label>
          <label>
            GPUs per worker
            <input
              type="number"
              min={0}
              max={8}
              value={jobGpus}
              onChange={(e) => onJobGpusChange(Number(e.target.value))}
            />
          </label>
        </div>
      </article>

      <article className="card">
        <h3>Transformer architecture (from scratch)</h3>
        <label>
          Preset
          <select value={preset} onChange={(e) => onPresetChange(e.target.value)}>
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
          maxPosition={selectedPreset?.config.architecture.max_position_embeddings ?? 512}
          paramEstimate={paramEstimate}
          onHiddenSizeChange={onHiddenSizeChange}
          onLayersChange={onLayersChange}
          onHeadsChange={onHeadsChange}
        />
        <div className="arch-grid arch-grid-compact">
          <label>
            Hidden size
            <input
              type="number"
              value={hiddenSize}
              onChange={(e) => onHiddenSizeChange(Number(e.target.value))}
            />
          </label>
          <label>
            Layers
            <input
              type="number"
              value={layers}
              onChange={(e) => onLayersChange(Number(e.target.value))}
            />
          </label>
          <label>
            Attention heads
            <input
              type="number"
              value={heads}
              onChange={(e) => onHeadsChange(Number(e.target.value))}
            />
          </label>
          <label>
            Training steps
            <input
              type="number"
              value={maxSteps}
              onChange={(e) => onMaxStepsChange(Number(e.target.value))}
            />
          </label>
        </div>
        <label>
          Experiment
          <select
            value={selectedExp ?? ""}
            onChange={(e) => onSelectedExpChange(Number(e.target.value))}
          >
            {experiments.map((e) => (
              <option key={e.id} value={e.id}>{e.name}</option>
            ))}
          </select>
        </label>
        <label>
          OpenAI model id
          <input value={openaiModelId} onChange={(e) => onOpenaiModelIdChange(e.target.value)} />
        </label>
        <label>
          Job name
          <input value={jobName} onChange={(e) => onJobNameChange(e.target.value)} />
        </label>
        <button type="button" onClick={onCreateJob} disabled={!selectedExp}>
          Start model task
        </button>
      </article>
    </div>
  );
}
