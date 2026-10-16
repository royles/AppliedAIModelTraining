type Props = {
  hiddenSize: number;
  layers: number;
  heads: number;
  vocabSize: number;
  maxPosition: number;
  paramEstimate: number | null;
  onHiddenSizeChange: (value: number) => void;
  onLayersChange: (value: number) => void;
  onHeadsChange: (value: number) => void;
};

function clampInt(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return min;
  return Math.min(max, Math.max(min, Math.trunc(value)));
}

export function TransformerArchitectureDiagram({
  hiddenSize,
  layers,
  heads,
  vocabSize,
  maxPosition,
  paramEstimate,
  onHiddenSizeChange,
  onLayersChange,
  onHeadsChange,
}: Props) {
  const headDim =
    heads > 0 && hiddenSize % heads === 0 ? hiddenSize / heads : null;
  const ffnInner = hiddenSize * 4;
  const headsValid = headDim !== null;

  return (
    <div className="transformer-diagram-wrap">
      <p className="hint diagram-caption">
        Left-to-right data flow · widths and depth update as you edit parameters
      </p>
      <div className="transformer-diagram-scroll">
        <svg
          className="transformer-diagram"
          viewBox="0 0 920 220"
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

          <g className="diagram-node" transform="translate(8, 72)">
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

          <line x1="96" y1="110" x2="118" y2="110" className="diagram-arrow" markerEnd="url(#arrow)" />

          <g className="diagram-node" transform="translate(120, 56)">
            <rect width="108" height="108" rx="8" />
            <text x="54" y="26" textAnchor="middle" className="diagram-title">
              Embedding
            </text>
            <text x="54" y="48" textAnchor="middle" className="diagram-sub">
              {vocabSize.toLocaleString()} × d
            </text>
            <foreignObject x="14" y="58" width="80" height="40">
              <label className="diagram-input-label">
                d (hidden)
                <input
                  type="number"
                  min={64}
                  max={8192}
                  value={hiddenSize}
                  onChange={(e) =>
                    onHiddenSizeChange(clampInt(Number(e.target.value), 64, 8192))
                  }
                />
              </label>
            </foreignObject>
          </g>

          <line x1="228" y1="110" x2="250" y2="110" className="diagram-arrow" markerEnd="url(#arrow)" />

          <g className="diagram-node diagram-node-muted" transform="translate(252, 72)">
            <rect width="96" height="76" rx="8" />
            <text x="48" y="28" textAnchor="middle" className="diagram-title">
              + Position
            </text>
            <text x="48" y="50" textAnchor="middle" className="diagram-sub">
              {maxPosition} × d
            </text>
          </g>

          <line x1="348" y1="110" x2="370" y2="110" className="diagram-arrow" markerEnd="url(#arrow)" />

          <g className="diagram-node diagram-node-stack" transform="translate(372, 24)">
            <rect width="248" height="172" rx="10" />
            <text x="124" y="22" textAnchor="middle" className="diagram-title">
              Transformer block
            </text>
            <text x="124" y="40" textAnchor="middle" className="diagram-sub">
              repeated ×
            </text>
            <foreignObject x="88" y="44" width="72" height="36">
              <label className="diagram-input-label diagram-input-inline">
                layers
                <input
                  type="number"
                  min={1}
                  max={128}
                  value={layers}
                  onChange={(e) =>
                    onLayersChange(clampInt(Number(e.target.value), 1, 128))
                  }
                />
              </label>
            </foreignObject>

            <rect x="16" y="88" width="100" height="68" rx="6" className="diagram-inner" />
            <text x="66" y="108" textAnchor="middle" className="diagram-inner-title">
              Multi-head
            </text>
            <text x="66" y="124" textAnchor="middle" className="diagram-inner-title">
              attention
            </text>
            <foreignObject x="22" y="128" width="88" height="32">
              <label className="diagram-input-label diagram-input-compact">
                heads
                <input
                  type="number"
                  min={1}
                  max={128}
                  value={heads}
                  onChange={(e) =>
                    onHeadsChange(clampInt(Number(e.target.value), 1, 128))
                  }
                />
              </label>
            </foreignObject>

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

          <line x1="620" y1="110" x2="642" y2="110" className="diagram-arrow" markerEnd="url(#arrow)" />

          <g className="diagram-node" transform="translate(644, 72)">
            <rect width="108" height="76" rx="8" />
            <text x="54" y="28" textAnchor="middle" className="diagram-title">
              LM head
            </text>
            <text x="54" y="50" textAnchor="middle" className="diagram-sub">
              d × vocab
            </text>
            <text x="54" y="66" textAnchor="middle" className="diagram-sub">
              logits
            </text>
          </g>

          <line x1="752" y1="110" x2="774" y2="110" className="diagram-arrow" markerEnd="url(#arrow)" />

          <g className="diagram-node" transform="translate(776, 80)">
            <rect width="88" height="60" rx="8" />
            <text x="44" y="36" textAnchor="middle" className="diagram-title">
              Output
            </text>
          </g>

          {layers > 1 && (
            <g className="diagram-stack-badge" transform="translate(372, 8)">
              <text x="124" y="0" textAnchor="middle">
                {layers} identical blocks stacked (depth)
              </text>
            </g>
          )}
        </svg>
      </div>
      {paramEstimate !== null && (
        <p className="hint diagram-params">
          ~{paramEstimate.toLocaleString()} trainable parameters (estimate)
        </p>
      )}
    </div>
  );
}
