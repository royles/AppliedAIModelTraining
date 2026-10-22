import { useId, useMemo } from "react";
import { formatCompactCount, transformerParamBreakdown } from "./transformerNetworkMetrics";

type ArchFamily = "gpt2" | "bert";
type InitMode = "from_scratch" | "pretrained";
type TaskType = "causal_lm" | "sequence_classification";

export type TransformerNetworkDiagramProps = {
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
  /** Side-by-side with geometry fields — tighter chrome, stats in the form column. */
  embedded?: boolean;
};

function familyLabel(family: ArchFamily, taskType: TaskType): string {
  if (family === "bert" || taskType === "sequence_classification") {
    return "Encoder (BERT-style)";
  }
  return "Decoder-only GPT";
}

/** Which layer indices to draw in the vertical stack (-1 = ellipsis). */
function layerDisplayIndices(layerCount: number): number[] {
  if (layerCount <= 0) {
    return [];
  }
  if (layerCount <= 10) {
    return Array.from({ length: layerCount }, (_, i) => i);
  }
  const mid = Math.floor(layerCount / 2);
  return [0, 1, 2, -1, mid, -1, layerCount - 3, layerCount - 2, layerCount - 1].filter(
    (v, i, a) => a.indexOf(v) === i,
  );
}

function headDisplayCount(heads: number): number {
  return Math.min(Math.max(heads, 1), 14);
}

export function TransformerNetworkDiagram(props: TransformerNetworkDiagramProps) {
  const {
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
    embedded = false,
  } = props;

  const uid = useId().replace(/:/g, "");
  const headDim = heads > 0 && hiddenSize % heads === 0 ? hiddenSize / heads : null;
  const headsValid = headDim !== null;
  const ffnInner = intermediateSize > 0 ? intermediateSize : hiddenSize * 4;
  const isClassifier = taskType === "sequence_classification" || family === "bert";
  const isCausal = !isClassifier;
  const fromScratch = initMode === "from_scratch";
  const archSummary = familyLabel(family, taskType);

  const breakdown = useMemo(
    () =>
      transformerParamBreakdown({
        vocabSize,
        hiddenSize,
        layers,
        heads,
        intermediateSize: ffnInner,
        maxPosition,
        tieWordEmbeddings,
      }),
    [vocabSize, hiddenSize, layers, heads, ffnInner, maxPosition, tieWordEmbeddings],
  );

  const layerIndices = layerDisplayIndices(layers);
  const heroLayer = layers <= 1 ? 0 : Math.floor(layers / 2);
  const headsDrawn = headDisplayCount(heads);
  const ffnBarMax = Math.max(hiddenSize, ffnInner);
  const ffnExpandW = 36 + Math.round((ffnInner / ffnBarMax) * 72);
  const ffnCoreW = 36 + Math.round((hiddenSize / ffnBarMax) * 72);

  const miniBlockH = 28;
  const miniGap = 6;
  const stackCount = layerIndices.length;
  const stackH = stackCount * miniBlockH + Math.max(0, stackCount - 1) * miniGap;
  const heroH = 200;
  const pad = 24;
  const W = 880;
  const H = pad + 56 + stackH + heroH + 88 + pad;

  const stackX = 320;
  const stackY = pad + 48;
  const heroX = 520;
  const heroY = stackY + Math.min(stackH * 0.35, stackH - 40);

  const embedCols = Math.min(12, Math.max(4, Math.round(hiddenSize / 64)));

  return (
    <div
      className={`transformer-network-viz${embedded ? " transformer-network-viz-embedded" : ""}`}
      role="figure"
      aria-label="Transformer network layout from your geometry"
    >
      {!embedded && (
        <div className="transformer-network-viz-header">
          <div>
            <p className="transformer-network-viz-title">
              {presetTitle ? `${presetTitle} · ` : ""}
              {archSummary}
              {fromScratch ? " · random init" : " · fine-tune"}
              {useLora ? " · LoRA" : ""}
            </p>
            {!fromScratch && baseModel && (
              <p className="hint transformer-network-viz-checkpoint">
                Checkpoint: <code>{baseModel}</code>
              </p>
            )}
          </div>
          <dl className="transformer-network-viz-stats">
            <div>
              <dt>d</dt>
              <dd>{hiddenSize}</dd>
            </div>
            <div>
              <dt>L</dt>
              <dd>{layers}</dd>
            </div>
            <div>
              <dt>H</dt>
              <dd>{heads}</dd>
            </div>
            <div>
              <dt>d_head</dt>
              <dd>{headsValid ? headDim : "—"}</dd>
            </div>
            <div>
              <dt>FFN</dt>
              <dd>
                {hiddenSize}→{ffnInner}
              </dd>
            </div>
            <div>
              <dt>Vocab</dt>
              <dd>{formatCompactCount(vocabSize)}</dd>
            </div>
            <div>
              <dt>Ctx</dt>
              <dd>{maxPosition}</dd>
            </div>
            <div>
              <dt>Params</dt>
              <dd>{formatCompactCount(paramEstimate ?? breakdown.total)}</dd>
            </div>
          </dl>
        </div>
      )}
      {embedded && (
        <p className="hint transformer-network-viz-embedded-lead">
          {archSummary}
          {presetTitle ? ` · ${presetTitle}` : ""}
          {!fromScratch && baseModel ? ` · ${baseModel}` : ""}
        </p>
      )}

      <div className="transformer-network-viz-canvas">
        <svg
          className="transformer-network-viz-svg"
          viewBox={`0 0 ${W} ${H}`}
          preserveAspectRatio="xMidYMid meet"
        >
          <defs>
            <linearGradient id={`${uid}-bg`} x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stopColor="#0f172a" />
              <stop offset="100%" stopColor="#1e293b" />
            </linearGradient>
            <linearGradient id={`${uid}-flow`} x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="#22d3ee" stopOpacity="0.15" />
              <stop offset="50%" stopColor="#818cf8" stopOpacity="0.35" />
              <stop offset="100%" stopColor="#38bdf8" stopOpacity="0.15" />
            </linearGradient>
            <linearGradient id={`${uid}-residual`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#fb923c" />
              <stop offset="100%" stopColor="#f97316" />
            </linearGradient>
            <filter id={`${uid}-glow`} x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="2" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
            <marker id={`${uid}-arrow`} markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
              <path d="M0,0 L8,4 L0,8 Z" fill="#67e8f9" />
            </marker>
          </defs>

          <rect x="0" y="0" width={W} height={H} rx="12" fill={`url(#${uid}-bg)`} />

          {/* Input → embeddings */}
          <g transform={`translate(${pad}, ${stackY + stackH / 2 - 40})`}>
            <text x="0" y="0" className="tn-label">
              Token ids
            </text>
            <rect x="0" y="8" width="56" height="48" rx="6" className="tn-node" />
            <text x="28" y="38" textAnchor="middle" className="tn-sublabel">
              seq
            </text>

            <path d="M 58 32 H 78" className="tn-arrow" markerEnd={`url(#${uid}-arrow)`} />

            <g transform="translate(82, 0)">
              <text x="0" y="0" className="tn-label">
                Token embed
              </text>
              <text x="0" y="14" className="tn-mono">
                {vocabSize.toLocaleString()} × {hiddenSize}
              </text>
              <g transform="translate(0, 22)">
                {Array.from({ length: embedCols }, (_, c) => (
                  <rect
                    key={c}
                    x={c * 7}
                    y="0"
                    width="5"
                    height="36"
                    rx="1"
                    className="tn-embed-col"
                    style={{ opacity: 0.35 + (c / embedCols) * 0.55 }}
                  />
                ))}
              </g>
            </g>

            <path d="M 182 32 H 202" className="tn-arrow" markerEnd={`url(#${uid}-arrow)`} />

            <g transform="translate(206, 8)">
              <rect width="72" height="48" rx="6" className="tn-node tn-node-muted" />
              <text x="36" y="22" textAnchor="middle" className="tn-sublabel">
                + position
              </text>
              <text x="36" y="38" textAnchor="middle" className="tn-mono">
                {maxPosition}×d
              </text>
            </g>

            <path
              d={`M 278 32 C 300 32, 300 ${stackY + stackH / 2 - (stackY + stackH / 2 - 40)}, ${stackX - 8} ${stackY + stackH / 2}`}
              className="tn-arrow"
              markerEnd={`url(#${uid}-arrow)`}
              fill="none"
            />
          </g>

          {/* Layer stack */}
          <text x={stackX + 70} y={stackY - 8} textAnchor="middle" className="tn-section">
            {layers} transformer block{layers === 1 ? "" : "s"} (L={layers})
          </text>
          {layerIndices.map((layerIdx, i) => {
            const y = stackY + i * (miniBlockH + miniGap);
            if (layerIdx === -1) {
              return (
                <g key={`ell-${i}`} transform={`translate(${stackX}, ${y})`}>
                  <text x="70" y={miniBlockH / 2 + 4} textAnchor="middle" className="tn-ellipsis">
                    ⋮
                  </text>
                </g>
              );
            }
            const isHero = layerIdx === heroLayer;
            return (
              <g key={layerIdx} transform={`translate(${stackX}, ${y})`}>
                <rect
                  width="140"
                  height={miniBlockH}
                  rx="6"
                  className={isHero ? "tn-layer tn-layer-active" : "tn-layer"}
                />
                <text x="10" y="17" className="tn-sublabel">
                  Layer {layerIdx}
                </text>
                <text x="10" y="26" className="tn-mini-flow">
                  MHA → FFN
                </text>
                {isHero && (
                  <text x="130" y="18" textAnchor="end" className="tn-hero-tag">
                    expanded →
                  </text>
                )}
                {!isHero && (
                  <rect
                    x="88"
                    y="8"
                    width="44"
                    height="12"
                    rx="2"
                    className="tn-mini-bar"
                    style={{ width: `${Math.min(44, 12 + layers * 2)}px` }}
                  />
                )}
              </g>
            );
          })}

          {/* Hero block cutaway */}
          <g transform={`translate(${heroX}, ${heroY})`} filter={`url(#${uid}-glow)`}>
            <rect width="320" height={heroH} rx="10" className="tn-hero-frame" />
            <text x="160" y="22" textAnchor="middle" className="tn-section">
              Block {heroLayer} internals (d={hiddenSize})
            </text>

            {/* Residual highway */}
            <rect x="16" y="36" width="10" height={heroH - 52} rx="4" fill={`url(#${uid}-residual)`} opacity="0.85" />
            <text x="21" y={heroH - 18} textAnchor="middle" className="tn-residual-label" transform={`rotate(-90 21 ${heroH - 18})`}>
              residual stream
            </text>

            {/* LN + MHA */}
            <g transform="translate(40, 40)">
              <rect width="48" height="24" rx="4" className="tn-submod" />
              <text x="24" y="16" textAnchor="middle" className="tn-sublabel">
                LayerNorm
              </text>
            </g>

            <g transform="translate(100, 34)">
              <rect width="200" height="92" rx="8" className="tn-submod tn-submod-attn" />
              <text x="100" y="18" textAnchor="middle" className="tn-sublabel">
                {isCausal ? "Causal multi-head attention" : "Multi-head attention"}
              </text>
              <text x="100" y="32" textAnchor="middle" className="tn-mono">
                Q,K,V: d×d · out proj d×d · {formatCompactCount(breakdown.perLayerAttention)} params/layer
              </text>
              <g transform="translate(12, 42)">
                {Array.from({ length: headsDrawn }, (_, h) => (
                  <g key={h} transform={`translate(${h * 13}, 0)`}>
                    <rect width="11" height="40" rx="2" className="tn-head" />
                    {isCausal && (
                      <path
                        d="M 0 40 L 11 40 L 11 0"
                        className="tn-causal-mask"
                        opacity="0.35"
                      />
                    )}
                  </g>
                ))}
                {heads > headsDrawn && (
                  <text x={headsDrawn * 13 + 4} y="24" className="tn-mono">
                    +{heads - headsDrawn} heads
                  </text>
                )}
              </g>
              <text x="100" y="88" textAnchor="middle" className="tn-mono">
                H={heads}
                {headsValid ? ` · d_head=${headDim}` : " · fix hidden÷heads"}
              </text>
            </g>

            {/* Residual add 1 */}
            <circle cx="34" cy="138" r="8" className="tn-add" />
            <text x="34" y="142" textAnchor="middle" className="tn-add-text">
              +
            </text>

            {/* LN + FFN */}
            <g transform="translate(40, 152)">
              <rect width="48" height="24" rx="4" className="tn-submod" />
              <text x="24" y="16" textAnchor="middle" className="tn-sublabel">
                LayerNorm
              </text>
            </g>

            <g transform="translate(100, 138)">
              <rect width="200" height="56" rx="8" className="tn-submod tn-submod-ffn" />
              <text x="100" y="18" textAnchor="middle" className="tn-sublabel">
                Feed-forward (MLP)
              </text>
              <g transform="translate(24, 26)">
                <rect width={ffnCoreW} height="22" rx="4" className="tn-ffn-bar" />
                <text x={ffnCoreW / 2} y="15" textAnchor="middle" className="tn-mono">
                  d
                </text>
                <path d={`M ${ffnCoreW + 4} 11 H ${ffnCoreW + 16}`} className="tn-arrow-thin" markerEnd={`url(#${uid}-arrow)`} />
                <rect x={ffnCoreW + 18} width={ffnExpandW} height="22" rx="4" className="tn-ffn-bar tn-ffn-expand" />
                <text x={ffnCoreW + 18 + ffnExpandW / 2} y="15" textAnchor="middle" className="tn-mono">
                  {ffnInner}
                </text>
                <path
                  d={`M ${ffnCoreW + 22 + ffnExpandW} 11 H ${ffnCoreW + 34 + ffnExpandW}`}
                  className="tn-arrow-thin"
                  markerEnd={`url(#${uid}-arrow)`}
                />
                <rect
                  x={ffnCoreW + 36 + ffnExpandW}
                  width={ffnCoreW}
                  height="22"
                  rx="4"
                  className="tn-ffn-bar"
                />
              </g>
              <text x="100" y="52" textAnchor="middle" className="tn-mono">
                {formatCompactCount(breakdown.perLayerFfn)} params/layer (2× d×ffn)
              </text>
            </g>

            <circle cx="34" cy="178" r="8" className="tn-add" />
            <text x="34" y="182" textAnchor="middle" className="tn-add-text">
              +
            </text>

            {useLora && (
              <text x="160" y={heroH - 8} textAnchor="middle" className="tn-lora">
                LoRA adapters on attention &amp; MLP projections
              </text>
            )}
          </g>

          {/* Output head */}
          <g transform={`translate(${pad}, ${stackY + stackH + 24})`}>
            <path
              d={`M ${stackX + 70} ${stackY + stackH} C ${stackX + 70} ${stackY + stackH + 20}, 120 ${stackY + stackH + 40}, 120 ${stackY + stackH + 48}`}
              className="tn-arrow"
              fill="none"
              markerEnd={`url(#${uid}-arrow)`}
            />
            <rect x="0" y="0" width="100" height="52" rx="8" className="tn-node" />
            <text x="50" y="22" textAnchor="middle" className="tn-sublabel">
              {isClassifier ? "Pool + classifier" : "LM head"}
            </text>
            <text x="50" y="38" textAnchor="middle" className="tn-mono">
              {isClassifier ? "d × labels" : tieWordEmbeddings ? "tied to embed" : `d × ${formatCompactCount(vocabSize)}`}
            </text>
            <path d="M 102 26 H 130" className="tn-arrow" markerEnd={`url(#${uid}-arrow)`} />
            <rect x="132" y="6" width="88" height="40" rx="8" className="tn-node tn-node-out" />
            <text x="176" y="32" textAnchor="middle" className="tn-sublabel">
              {isClassifier ? "logits" : "next-token logits"}
            </text>
          </g>

          {/* Width annotation */}
          <rect x={pad} y={H - pad - 28} width={W - pad * 2} height="20" rx="4" fill={`url(#${uid}-flow)`} />
          <text x={W / 2} y={H - pad - 14} textAnchor="middle" className="tn-footer-note">
            Hidden width d={hiddenSize} flows through every block · {layers}×(attn + FFN) · ~
            {formatCompactCount(breakdown.allLayers)} params in blocks
          </text>
        </svg>
      </div>

      {!embedded && (
        <dl className="transformer-network-viz-legend">
          <div>
            <dt>Embeddings</dt>
            <dd>
              {formatCompactCount(breakdown.tokenEmbedding)} (V×d)
              {!fromScratch && " · from checkpoint"}
            </dd>
          </div>
          <div>
            <dt>Each layer</dt>
            <dd>
              attn {formatCompactCount(breakdown.perLayerAttention)} + FFN{" "}
              {formatCompactCount(breakdown.perLayerFfn)}
            </dd>
          </div>
          <div>
            <dt>LM head</dt>
            <dd>{tieWordEmbeddings ? "weight-tied (0 extra)" : formatCompactCount(breakdown.lmHead)}</dd>
          </div>
        </dl>
      )}
    </div>
  );
}
