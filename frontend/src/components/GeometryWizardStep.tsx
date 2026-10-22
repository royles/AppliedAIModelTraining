import { ValidatedNumberField } from "../ValidatedNumberField";

type Props = {
  initMode: "from_scratch" | "pretrained";
  hiddenSize: number;
  layers: number;
  heads: number;
  maxPosition: number;
  vocabSize: number;
  intermediateSize: number;
  paramEstimate: number | null;
  onHiddenSizeChange: (n: number) => void;
  onLayersChange: (n: number) => void;
  onHeadsChange: (n: number) => void;
  onMaxPositionChange: (n: number) => void;
};

export function GeometryWizardStep({
  initMode,
  hiddenSize,
  layers,
  heads,
  maxPosition,
  vocabSize,
  intermediateSize,
  paramEstimate,
  onHiddenSizeChange,
  onLayersChange,
  onHeadsChange,
  onMaxPositionChange,
}: Props) {
  const geometryEditable = initMode === "from_scratch";
  const headsValid = heads > 0 && hiddenSize % heads === 0;

  return (
    <div className="wizard-step-compact">
      <p className="hint architecture-section-lead">
        {geometryEditable
          ? "Hidden size must divide evenly by attention heads."
          : "Fine-tune uses preset geometry; switch Mode to from-scratch to edit."}
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
          <span className="architecture-field-label">Heads (H)</span>
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
      <dl className="architecture-metrics architecture-metrics-inline">
        <div>
          <dt>d_head</dt>
          <dd>{headsValid ? hiddenSize / heads : "—"}</dd>
        </div>
        <div>
          <dt>FFN</dt>
          <dd>{intermediateSize > 0 ? intermediateSize : hiddenSize * 4}</dd>
        </div>
        <div>
          <dt>Vocab</dt>
          <dd>{vocabSize.toLocaleString()}</dd>
        </div>
        {paramEstimate != null && geometryEditable && (
          <div>
            <dt>Params</dt>
            <dd>{paramEstimate.toLocaleString()}</dd>
          </div>
        )}
      </dl>
    </div>
  );
}
