/** Parameter counts aligned with backend `TransformerArchitectureConfig.parameter_count_estimate`. */
export function transformerParamBreakdown(args: {
  vocabSize: number;
  hiddenSize: number;
  layers: number;
  heads: number;
  intermediateSize: number;
  maxPosition: number;
  tieWordEmbeddings: boolean;
}): {
  tokenEmbedding: number;
  positionEmbedding: number;
  perLayerAttention: number;
  perLayerFfn: number;
  perLayerTotal: number;
  allLayers: number;
  lmHead: number;
  total: number;
} {
  const { vocabSize: v, hiddenSize: h, layers: l } = args;
  const ffn = args.intermediateSize > 0 ? args.intermediateSize : h * 4;
  const tokenEmbedding = v * h;
  const positionEmbedding = args.maxPosition * h;
  const perLayerAttention = 4 * h * h;
  const perLayerFfn = 2 * h * ffn;
  const perLayerTotal = perLayerAttention + perLayerFfn;
  const allLayers = l * perLayerTotal;
  const lmHead = args.tieWordEmbeddings ? 0 : v * h;
  const total = tokenEmbedding + allLayers + lmHead;
  return {
    tokenEmbedding,
    positionEmbedding,
    perLayerAttention,
    perLayerFfn,
    perLayerTotal,
    allLayers,
    lmHead,
    total,
  };
}

export function formatCompactCount(n: number): string {
  if (n >= 1_000_000_000) {
    return `${(n / 1_000_000_000).toFixed(2)}B`;
  }
  if (n >= 1_000_000) {
    return `${(n / 1_000_000).toFixed(2)}M`;
  }
  if (n >= 1_000) {
    return `${(n / 1_000).toFixed(1)}K`;
  }
  return String(n);
}
