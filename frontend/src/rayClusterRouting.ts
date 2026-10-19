/** Orthogonal SVG paths for the Ray cluster diagram (avoid crossing node cards). */

export type LinkAnchor = {
  key: string;
  healthy: boolean;
  sx: number;
  sy: number;
  ex: number;
  ey: number;
};

export type RoutedLink = {
  key: string;
  healthy: boolean;
  d: string;
};

const STUB = 10;

function pathFromPoints(points: [number, number][]): string {
  if (points.length === 0) {
    return "";
  }
  return points
    .map((p, i) => (i === 0 ? `M ${p[0].toFixed(1)} ${p[1].toFixed(1)}` : `L ${p[0].toFixed(1)} ${p[1].toFixed(1)}`))
    .join(" ");
}

/**
 * Route via a horizontal corridor above (or below) all nodes, with per-link lane offsets.
 */
export function routeClusterLinks(
  anchors: LinkAnchor[],
  bounds: { minTop: number; maxBottom: number },
): RoutedLink[] {
  if (anchors.length === 0) {
    return [];
  }

  const topCorridor = bounds.minTop - 20;
  const bottomCorridor = bounds.maxBottom + 20;
  const useTop = true;
  const baseY = useTop ? topCorridor : bottomCorridor;

  const laneStep = 7;
  const laneCount = Math.max(1, Math.ceil(anchors.length / 3));

  return anchors.map((a, index) => {
    const lane = (index % laneCount) * laneStep - ((laneCount - 1) * laneStep) / 2;
    const cy = baseY + lane;
    const forward = a.ex >= a.sx;

    let points: [number, number][];
    if (forward) {
      points = [
        [a.sx, a.sy],
        [a.sx + STUB, a.sy],
        [a.sx + STUB, cy],
        [a.ex - STUB, cy],
        [a.ex - STUB, a.ey],
        [a.ex, a.ey],
      ];
    } else {
      // Rare reverse edge: dip below the row
      const by = bottomCorridor + Math.abs(lane);
      points = [
        [a.sx, a.sy],
        [a.sx + STUB, a.sy],
        [a.sx + STUB, by],
        [a.ex - STUB, by],
        [a.ex - STUB, a.ey],
        [a.ex, a.ey],
      ];
    }

    return { key: a.key, healthy: a.healthy, d: pathFromPoints(points) };
  });
}

export function routedLinksEqual(a: RoutedLink[], b: RoutedLink[]): boolean {
  if (a.length !== b.length) {
    return false;
  }
  for (let i = 0; i < a.length; i += 1) {
    if (a[i].key !== b[i].key || a[i].healthy !== b[i].healthy || a[i].d !== b[i].d) {
      return false;
    }
  }
  return true;
}
