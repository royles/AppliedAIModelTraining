/** Orthogonal SVG paths for the Ray cluster diagram (avoid crossing node cards). */

export type LinkAnchor = {
  key: string;
  healthy: boolean;
  /** Source: right edge, top corner */
  sx: number;
  sourceTop: number;
  /** Target: left edge (for final hop along top border) */
  ex: number;
  ey: number;
  /** Vertical drop column (target horizontal center) */
  tx: number;
  /** Target top edge (canvas coordinates) */
  targetTop: number;
};

export type RoutedLink = {
  key: string;
  healthy: boolean;
  d: string;
};

const STUB = 8;
const CORRIDOR_ABOVE = 14;

function pathFromPoints(points: [number, number][]): string {
  if (points.length === 0) {
    return "";
  }
  return points
    .map((p, i) =>
      i === 0 ? `M ${p[0].toFixed(1)} ${p[1].toFixed(1)}` : `L ${p[0].toFixed(1)} ${p[1].toFixed(1)}`,
    )
    .join(" ");
}

/**
 * Route only through the link band and the gap above nodes — never through card interiors.
 */
export function routeClusterLinks(
  anchors: LinkAnchor[],
  bounds: { linkBandBottom: number; minNodeTop: number },
): RoutedLink[] {
  if (anchors.length === 0) {
    return [];
  }

  const laneStep = 3;
  const maxLane = Math.min(12, Math.max(0, anchors.length - 1));
  const corridorTop = 6;

  return anchors.map((a, index) => {
    const lane = maxLane > 0 ? (index % (maxLane + 1)) * laneStep : 0;
    const yCorridor = Math.min(bounds.linkBandBottom - CORRIDOR_ABOVE - lane, bounds.minNodeTop - 6);
    const yCorridorClamped = Math.max(corridorTop + lane, yCorridor);

    const dropEndY = Math.min(a.targetTop, bounds.minNodeTop);

    let points: [number, number][];
    if (a.ex >= a.sx) {
      points = [
        [a.sx, a.sourceTop],
        [a.sx + STUB, a.sourceTop],
        [a.sx + STUB, yCorridorClamped],
        [a.tx, yCorridorClamped],
        [a.tx, dropEndY],
      ];
      if (Math.abs(a.ex - a.tx) > 2) {
        points.push([a.ex, dropEndY]);
      }
    } else {
      const yLow = bounds.minNodeTop + 24 + lane;
      points = [
        [a.sx, a.sourceTop],
        [a.sx + STUB, a.sourceTop],
        [a.sx + STUB, yLow],
        [a.tx, yLow],
        [a.tx, dropEndY],
      ];
      if (Math.abs(a.ex - a.tx) > 2) {
        points.push([a.ex, dropEndY]);
      }
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
