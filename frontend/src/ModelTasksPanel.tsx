import type { Project, TrainingDatasetConfig, TransformerPreset, TokenizerBuildConfig } from "./api";
import { DatasetConfigFields } from "./DatasetConfigFields";
import { FromScratchConfigFields } from "./FromScratchConfigFields";
import { TrainingRecipeChecklist } from "./TrainingRecipeChecklist";
import { TransformerArchitectureDiagram } from "./TransformerArchitectureDiagram";
import { useState } from "react";

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

  const [recipeReady, setRecipeReady] = useState(false);

  return (
    <div className="settings-page">
      <header className="main-panel-header">
        <h2>Model Tasks</h2>
        <p className="hint">
          Configure the training recipe below; start the task when every step is checked.
        </p>
      </header>

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
        onReadyChange={setRecipeReady}
      />

      <article className="card">
        <p className="recipe-step-label">Recipe step: Project</p>
        <h3>Project</h3>
        <label>
          New project name
          <input value={projectName} onChange={(e) => onProjectNameChange(e.target.value)} />
        </label>
        <button type="button" onClick={onCreateProject}>Create project</button>
      </article>

      <article className="card">
        <p className="recipe-step-label">Recipe step: Ray / CDSW compute</p>
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

      <div className="recipe-step-label-wrap">
        <p className="recipe-step-label">Recipe steps: Training mode · Tokenizer</p>
      </div>
      <FromScratchConfigFields
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

      <div className="recipe-step-label-wrap">
        <p className="recipe-step-label">Recipe step: Training dataset</p>
      </div>
      <DatasetConfigFields value={dataset} onChange={onDatasetChange} />

      <article className="card">
        <p className="recipe-step-label">
          Recipe steps: Transformer geometry · Optimization · Task metadata
        </p>
        <h3>
          Transformer architecture
          {initMode === "from_scratch" ? " (from scratch)" : " (fine-tune)"}
        </h3>
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
          <label>
            Max positions
            <input
              type="number"
              min={128}
              value={maxPosition}
              onChange={(e) => onMaxPositionChange(Number(e.target.value))}
            />
          </label>
        </div>
        <label>
          Project
          <select
            value={selectedProjectId ?? ""}
            onChange={(e) => onSelectedProjectIdChange(Number(e.target.value))}
          >
            {projects.map((p) => (
              <option key={p.id} value={p.id}>{p.name}</option>
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
        <button type="button" onClick={onCreateJob} disabled={!recipeReady}>
          Start model task
        </button>
        {!recipeReady && (
          <p className="hint">Complete all recipe steps before starting.</p>
        )}
      </article>
    </div>
  );
}
