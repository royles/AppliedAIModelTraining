/** Orthogonal SVG paths for the two-row Ray cluster diagram. */

export type NodeRow = "control" | "worker";

export type LinkAnchor = {
  key: string;
  healthy: boolean;
  sourceRow: NodeRow;
  targetRow: NodeRow;
  /** Source right edge (legacy measure) */
  sx: number;
  scx: number;
  sourceTop: number;
  sourceBottom: number;
  ex: number;
  tx: number;
  targetTop: number;
  targetBottom: number;
};

export type RoutedLink = {
  key: string;
  healthy: boolean;
  d: string;
};

export type LayoutBounds = {
  linkBandTop: number;
  linkBandBottom: number;
  controlRowTop: number;
  controlRowBottom: number;
  gutterTop: number;
  gutterBottom: number;
  workerRowTop: number;
  workerRowBottom: number;
};

const LINK_PAD = 3;

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

/** Horizontal lane in the top link band (strictly above control cards). */
function controlCorridorY(bounds: LayoutBounds, lane: number): number {
  const mid = (bounds.linkBandTop + bounds.linkBandBottom) / 2;
  const maxY = bounds.controlRowTop - LINK_PAD - lane;
  return Math.min(mid + lane * 1.5, maxY);
}

/** Lane in the gutter between control and worker rows. */
function gutterCorridorY(bounds: LayoutBounds, lane: number): number {
  const mid = (bounds.gutterTop + bounds.gutterBottom) / 2;
  const half = Math.max(3, (bounds.gutterBottom - bounds.gutterTop) / 2 - LINK_PAD);
  const spread = Math.min(half, lane * 1.25);
  return mid + (lane % 2 === 0 ? spread : -spread);
}

/** Lane in the gap above worker cards (below gutter). */
function workerApproachY(bounds: LayoutBounds, lane: number): number {
  const maxY = bounds.workerRowTop - LINK_PAD - lane;
  const minY = bounds.gutterBottom + LINK_PAD + lane;
  return Math.max(minY, Math.min(maxY, (minY + maxY) / 2));
}

function routeSameRow(a: LinkAnchor, bounds: LayoutBounds, lane: number): [number, number][] {
  const isWorker = a.sourceRow === "worker";
  const exitY = a.sourceTop;
  const enterY = a.targetTop;

  if (a.tx >= a.scx) {
    const yC = isWorker ? workerApproachY(bounds, lane) : controlCorridorY(bounds, lane);
    return [
      [a.scx, exitY],
      [a.scx, yC],
      [a.tx, yC],
      [a.tx, enterY],
    ];
  }

  const yC = isWorker ? bounds.workerRowBottom + LINK_PAD + lane : gutterCorridorY(bounds, lane);
  return [
    [a.scx, exitY],
    [a.scx, yC],
    [a.tx, yC],
    [a.tx, enterY],
  ];
}

function routeControlToWorker(a: LinkAnchor, bounds: LayoutBounds, lane: number): [number, number][] {
  const yG = gutterCorridorY(bounds, lane);
  return [
    [a.scx, a.sourceBottom],
    [a.scx, yG],
    [a.tx, yG],
    [a.tx, a.targetTop],
  ];
}

function routeWorkerToControl(a: LinkAnchor, bounds: LayoutBounds, lane: number): [number, number][] {
  const yG = gutterCorridorY(bounds, lane);
  return [
    [a.scx, a.sourceTop],
    [a.scx, yG],
    [a.tx, yG],
    [a.tx, a.targetBottom],
  ];
}

export function routeClusterLinks(anchors: LinkAnchor[], bounds: LayoutBounds): RoutedLink[] {
  if (anchors.length === 0) {
    return [];
  }

  const laneStep = 2;
  const maxLane = Math.min(8, Math.max(0, anchors.length - 1));

  return anchors.map((a, index) => {
    const lane = maxLane > 0 ? (index % (maxLane + 1)) * laneStep : 0;
    let points: [number, number][];
    if (a.sourceRow === a.targetRow) {
      points = routeSameRow(a, bounds, lane);
    } else if (a.sourceRow === "control" && a.targetRow === "worker") {
      points = routeControlToWorker(a, bounds, lane);
    } else {
      points = routeWorkerToControl(a, bounds, lane);
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
