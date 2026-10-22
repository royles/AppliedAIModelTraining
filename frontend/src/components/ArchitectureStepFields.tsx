import type { TransformerPreset } from "../api";
import { ModelTaskTrainingOptions } from "./ModelTaskTrainingOptions";
import { TransformerNetworkDiagram } from "../TransformerNetworkDiagram";
import { ValidatedNumberField } from "../ValidatedNumberField";
import type { PromptFormatSetting } from "../api";

type Props = {
  presets: TransformerPreset[];
  preset: string;
  onPresetChange: (name: string) => void;
  selectedPreset: TransformerPreset | undefined;
  taskType: "causal_lm" | "sequence_classification";
  onTaskTypeChange: (t: "causal_lm" | "sequence_classification") => void;
  promptFormat: PromptFormatSetting;
  onPromptFormatChange: (p: PromptFormatSetting) => void;
  initMode: "from_scratch" | "pretrained";
  archFamily: "gpt2" | "bert";
  hiddenSize: number;
  layers: number;
  heads: number;
  maxPosition: number;
  vocabSize: number;
  intermediateSize: number;
  tieWordEmbeddings: boolean;
  paramEstimate: number | null;
  baseModel?: string | null;
  useLora?: boolean;
  onHiddenSizeChange: (n: number) => void;
  onLayersChange: (n: number) => void;
  onHeadsChange: (n: number) => void;
  onMaxPositionChange: (n: number) => void;
};

export function ArchitectureStepFields({
  presets,
  preset,
  onPresetChange,
  selectedPreset,
  taskType,
  onTaskTypeChange,
  promptFormat,
  onPromptFormatChange,
  initMode,
  archFamily,
  hiddenSize,
  layers,
  heads,
  maxPosition,
  vocabSize,
  intermediateSize,
  tieWordEmbeddings,
  paramEstimate,
  baseModel,
  useLora,
  onHiddenSizeChange,
  onLayersChange,
  onHeadsChange,
  onMaxPositionChange,
}: Props) {
  const geometryEditable = initMode === "from_scratch";
  const headsValid = heads > 0 && hiddenSize % heads === 0;

  return (
    <div className="architecture-step">
      <section className="architecture-section" aria-labelledby="arch-preset-heading">
        <h4 id="arch-preset-heading" className="architecture-section-title">
          Preset & task
        </h4>
        <div className="architecture-form-grid architecture-form-grid-2">
          <label className="architecture-field">
            <span className="architecture-field-label">Preset</span>
            <select value={preset} onChange={(e) => onPresetChange(e.target.value)}>
              {presets.map((p) => (
                <option key={p.name} value={p.name}>
                  {p.title}
                </option>
              ))}
            </select>
            {selectedPreset?.description && (
              <span className="architecture-field-hint">{selectedPreset.description}</span>
            )}
          </label>
          <div className="architecture-field architecture-field-span">
            <ModelTaskTrainingOptions
              section="task"
              taskType={taskType}
              onTaskTypeChange={onTaskTypeChange}
              promptFormat={promptFormat}
              onPromptFormatChange={onPromptFormatChange}
            />
          </div>
        </div>
      </section>

      <section className="architecture-section" aria-labelledby="arch-diagram-heading">
        <h4 id="arch-diagram-heading" className="architecture-section-title">
          Block diagram
        </h4>
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
        />
      </section>

      <section className="architecture-section" aria-labelledby="arch-geometry-heading">
        <h4 id="arch-geometry-heading" className="architecture-section-title">
          Geometry
        </h4>
        <p className="hint architecture-section-lead">
          {geometryEditable
            ? "Set width, depth, and attention heads. Hidden size must divide evenly by head count."
            : "Values follow the preset checkpoint. Switch to from-scratch mode to edit geometry."}
        </p>
        <div className="architecture-form-grid">
          <label className="architecture-field">
            <span className="architecture-field-label">Hidden size (d)</span>
            <ValidatedNumberField
              value={hiddenSize}
              min={64}
              max={8192}
              disabled={!geometryEditable}
              onChange={onHiddenSizeChange}
            />
          </label>
          <label className="architecture-field">
            <span className="architecture-field-label">Layers (L)</span>
            <ValidatedNumberField
              value={layers}
              min={1}
              max={128}
              disabled={!geometryEditable}
              onChange={onLayersChange}
            />
          </label>
          <label className="architecture-field">
            <span className="architecture-field-label">Attention heads (H)</span>
            <ValidatedNumberField
              value={heads}
              min={1}
              max={128}
              disabled={!geometryEditable}
              onChange={onHeadsChange}
            />
          </label>
          <label className="architecture-field">
            <span className="architecture-field-label">Max positions</span>
            <ValidatedNumberField
              value={maxPosition}
              min={128}
              max={32768}
              disabled={!geometryEditable}
              onChange={onMaxPositionChange}
            />
          </label>
        </div>
        <dl className="architecture-metrics">
          <div>
            <dt>Head dimension</dt>
            <dd>{headsValid ? hiddenSize / heads : "—"}</dd>
          </div>
          <div>
            <dt>FFN inner</dt>
            <dd>{intermediateSize > 0 ? intermediateSize : hiddenSize * 4}</dd>
          </div>
          <div>
            <dt>Vocab (architecture)</dt>
            <dd>{vocabSize.toLocaleString()}</dd>
          </div>
          {paramEstimate != null && geometryEditable && (
            <div>
              <dt>Est. parameters</dt>
              <dd>{paramEstimate.toLocaleString()}</dd>
            </div>
          )}
        </dl>
        {!headsValid && (
          <p className="hint blueprint-warn" role="status">
            Hidden size ({hiddenSize}) is not divisible by attention heads ({heads}). Adjust one of
            these values.
          </p>
        )}
      </section>
    </div>
  );
}
