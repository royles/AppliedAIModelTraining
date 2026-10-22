type ArchFamily = "gpt2" | "bert";
type InitMode = "from_scratch" | "pretrained";
type TaskType = "causal_lm" | "sequence_classification";

type Props = {
  presetTitle?: string;
  family: ArchFamily;
  initMode: InitMode;
  taskType: TaskType;
  baseModel?: string | null;
  useLora?: boolean;
  hiddenSize: number;
  layers: number;
  heads: number;
  vocabSize: number;
  maxPosition: number;
  intermediateSize: number;
  tieWordEmbeddings: boolean;
  paramEstimate: number | null;
};

function familyLabel(family: ArchFamily, taskType: TaskType): string {
  if (family === "bert" || taskType === "sequence_classification") {
    return "Encoder (BERT-style)";
  }
  return "Decoder-only (GPT-2)";
}

export function TransformerArchitectureDiagram({
  presetTitle,
  family,
  initMode,
  taskType,
  baseModel,
  useLora,
  hiddenSize,
  layers,
  heads,
  vocabSize,
  maxPosition,
  intermediateSize,
  tieWordEmbeddings,
  paramEstimate,
}: Props) {
  const headDim =
    heads > 0 && hiddenSize % heads === 0 ? hiddenSize / heads : null;
  const ffnInner = intermediateSize > 0 ? intermediateSize : hiddenSize * 4;
  const headsValid = headDim !== null;
  const isClassifier = taskType === "sequence_classification" || family === "bert";
  const isCausal = !isClassifier;
  const fromScratch = initMode === "from_scratch";
  const stackGhosts = Math.min(Math.max(layers - 1, 0), 3);
  const attentionLabel = isCausal ? "Causal MHA" : "Bidirectional MHA";

  const archSummary = `${familyLabel(family, taskType)} · ${
    fromScratch ? "random init" : "pretrained"
  }${useLora ? " · LoRA" : ""}`;

  return (
    <div className="transformer-diagram-wrap">
      <p className="hint diagram-caption">
        {presetTitle ? `${presetTitle} · ` : ""}
        {archSummary}. Edit values in the geometry section below.
      </p>
      <div className="transformer-diagram-scroll">
        <svg
          className="transformer-diagram"
          viewBox="0 0 940 236"
          role="img"
          aria-label="Transformer architecture diagram"
        >
          <defs>
            <marker
              id="arrow"
              markerWidth="8"
              markerHeight="8"
              refX="7"
              refY="4"
              orient="auto"
            >
              <path d="M0,0 L8,4 L0,8 Z" fill="#5b8def" />
            </marker>
          </defs>

          <text x="470" y="16" textAnchor="middle" className="diagram-arch-badge">
            {archSummary}
          </text>

          {!fromScratch && baseModel && (
            <g className="diagram-node diagram-node-checkpoint" transform="translate(8, 28)">
              <rect width="120" height="44" rx="8" />
              <text x="60" y="20" textAnchor="middle" className="diagram-title">
                Checkpoint
              </text>
              <text x="60" y="36" textAnchor="middle" className="diagram-sub diagram-mono">
                {baseModel.length > 14 ? `${baseModel.slice(0, 12)}…` : baseModel}
              </text>
            </g>
          )}

          <g className="diagram-node" transform="translate(8, 88)">
            <rect width="88" height="76" rx="8" />
            <text x="44" y="28" textAnchor="middle" className="diagram-title">
              Tokens
            </text>
            <text x="44" y="48" textAnchor="middle" className="diagram-sub">
              seq
            </text>
            <text x="44" y="64" textAnchor="middle" className="diagram-sub">
              ≤ {maxPosition}
            </text>
          </g>

          <line x1="96" y1="126" x2="118" y2="126" className="diagram-arrow" markerEnd="url(#arrow)" />

          <g
            className={`diagram-node ${fromScratch ? "" : "diagram-node-muted"}`}
            transform="translate(120, 72)"
          >
            <rect width="108" height="108" rx="8" />
            <text x="54" y="26" textAnchor="middle" className="diagram-title">
              {fromScratch ? "Embedding" : "Embeddings"}
            </text>
            <text x="54" y="46" textAnchor="middle" className="diagram-sub">
              {fromScratch ? `${vocabSize.toLocaleString()} × d` : "from checkpoint"}
            </text>
            <text x="54" y="72" textAnchor="middle" className="diagram-sub diagram-mono">
              d = {hiddenSize}
            </text>
          </g>

          <line x1="228" y1="126" x2="250" y2="126" className="diagram-arrow" markerEnd="url(#arrow)" />

          <g className="diagram-node diagram-node-muted" transform="translate(252, 88)">
            <rect width="96" height="76" rx="8" />
            <text x="48" y="28" textAnchor="middle" className="diagram-title">
              + Position
            </text>
            <text x="48" y="50" textAnchor="middle" className="diagram-sub">
              {maxPosition} × d
            </text>
          </g>

          <line x1="348" y1="126" x2="370" y2="126" className="diagram-arrow" markerEnd="url(#arrow)" />

          {Array.from({ length: stackGhosts }, (_, i) => (
            <g
              key={`ghost-${i}`}
              className="diagram-layer-ghost"
              transform={`translate(${372 + (stackGhosts - i) * 6}, ${24 + (stackGhosts - i) * 5})`}
            >
              <rect width="248" height="172" rx="10" />
            </g>
          ))}

          <g className="diagram-node diagram-node-stack" transform="translate(372, 40)">
            <rect width="248" height="172" rx="10" />
            <text x="124" y="22" textAnchor="middle" className="diagram-title">
              Transformer block
            </text>
            {useLora && (
              <text x="124" y="38" textAnchor="middle" className="diagram-lora-badge">
                LoRA adapters on attention/MLP
              </text>
            )}
            <text x="124" y={useLora ? 58 : 44} textAnchor="middle" className="diagram-sub diagram-mono">
              × {layers} layers
            </text>

            <rect x="16" y="88" width="100" height="68" rx="6" className="diagram-inner" />
            <text x="66" y="108" textAnchor="middle" className="diagram-inner-title">
              {attentionLabel}
            </text>
            <text x="66" y="148" textAnchor="middle" className="diagram-sub diagram-mono">
              H = {heads}
            </text>

            <text x="178" y="108" textAnchor="middle" className="diagram-sub">
              d_head = {headsValid ? headDim : "—"}
            </text>
            {!headsValid && (
              <text x="178" y="124" textAnchor="middle" className="diagram-warn">
                hidden ÷ heads
              </text>
            )}

            <line x1="124" y1="88" x2="124" y2="156" className="diagram-inner-arrow" />
            <text x="124" y="82" textAnchor="middle" className="diagram-sub">
              ↓ + norm
            </text>

            <rect x="132" y="132" width="100" height="52" rx="6" className="diagram-inner" />
            <text x="182" y="152" textAnchor="middle" className="diagram-inner-title">
              FFN
            </text>
            <text x="182" y="168" textAnchor="middle" className="diagram-sub">
              d → {ffnInner} → d
            </text>
          </g>

          <line x1="620" y1="126" x2="642" y2="126" className="diagram-arrow" markerEnd="url(#arrow)" />

          <g className="diagram-node" transform="translate(644, 88)">
            <rect width="108" height="76" rx="8" />
            <text x="54" y="28" textAnchor="middle" className="diagram-title">
              {isClassifier ? "Class head" : "LM head"}
            </text>
            <text x="54" y="48" textAnchor="middle" className="diagram-sub">
              {isClassifier ? "d × labels" : tieWordEmbeddings ? "tied to embed" : "d × vocab"}
            </text>
            <text x="54" y="64" textAnchor="middle" className="diagram-sub">
              {isClassifier ? "logits" : "next-token"}
            </text>
          </g>

          <line x1="752" y1="126" x2="774" y2="126" className="diagram-arrow" markerEnd="url(#arrow)" />

          <g className="diagram-node" transform="translate(776, 96)">
            <rect width="88" height="60" rx="8" />
            <text x="44" y="36" textAnchor="middle" className="diagram-title">
              {isClassifier ? "Labels" : "Output"}
            </text>
          </g>

          {layers > 1 && (
            <g className="diagram-stack-badge" transform="translate(372, 224)">
              <text x="124" y="0" textAnchor="middle">
                {layers} stacked blocks (depth L={layers})
              </text>
            </g>
          )}
        </svg>
      </div>
      {paramEstimate !== null && fromScratch && (
        <p className="hint diagram-params">
          ~{paramEstimate.toLocaleString()} trainable parameters (estimate)
        </p>
      )}
      {!fromScratch && (
        <p className="hint diagram-params">
          Geometry follows the selected preset checkpoint; fine-tune updates weights (full or LoRA).
        </p>
      )}
    </div>
  );
}
