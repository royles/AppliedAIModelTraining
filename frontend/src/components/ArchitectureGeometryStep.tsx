import { ValidatedNumberField } from "../ValidatedNumberField";
import { TransformerNetworkDiagram } from "../TransformerNetworkDiagram";

type Props = {
  initMode: "from_scratch" | "pretrained";
  presetTitle?: string;
  family: "gpt2" | "bert";
  taskType: "causal_lm" | "sequence_classification";
  baseModel?: string | null;
  useLora?: boolean;
  hiddenSize: number;
  layers: number;
  heads: number;
  maxPosition: number;
  vocabSize: number;
  intermediateSize: number;
  tieWordEmbeddings: boolean;
  paramEstimate: number | null;
  onHiddenSizeChange: (n: number) => void;
  onLayersChange: (n: number) => void;
  onHeadsChange: (n: number) => void;
  onMaxPositionChange: (n: number) => void;
};

export function ArchitectureGeometryStep({
  initMode,
  presetTitle,
  family,
  taskType,
  baseModel,
  useLora,
  hiddenSize,
  layers,
  heads,
  maxPosition,
  vocabSize,
  intermediateSize,
  tieWordEmbeddings,
  paramEstimate,
  onHiddenSizeChange,
  onLayersChange,
  onHeadsChange,
  onMaxPositionChange,
}: Props) {
  const geometryEditable = initMode === "from_scratch";
  const headsValid = heads > 0 && hiddenSize % heads === 0;

  return (
    <div className="architecture-geometry-blend">
      <section className="architecture-geometry-blend-fields" aria-labelledby="arch-geometry-fields">
        <h4 id="arch-geometry-fields" className="architecture-section-title">
          Geometry
        </h4>
        <p className="hint architecture-section-lead">
          {geometryEditable
            ? "Width, depth, and context — the diagram updates as you edit."
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
          {paramEstimate != null && (
            <div>
              <dt>Params</dt>
              <dd>{paramEstimate.toLocaleString()}</dd>
            </div>
          )}
        </dl>
      </section>

      <section className="architecture-geometry-blend-viz" aria-labelledby="arch-network-viz">
        <h4 id="arch-network-viz" className="architecture-section-title">
          Network layout
        </h4>
        <TransformerNetworkDiagram
          embedded
          presetTitle={presetTitle}
          family={family}
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
    </div>
  );
}
