/** Orthogonal SVG paths for the two-row Ray cluster diagram. */

export type NodeRow = "control" | "worker";

export type LinkAnchor = {
  key: string;
  healthy: boolean;
  sourceRow: NodeRow;
  targetRow: NodeRow;
  /** Source right edge (same-row) or center x (cross-row exit) */
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
  linkBandBottom: number;
  controlRowTop: number;
  controlRowBottom: number;
  /** Vertical band between control and worker rows (links run here). */
  gutterTop: number;
  gutterBottom: number;
  workerRowTop: number;
  workerRowBottom: number;
};

const STUB = 10;

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

function sourceMidY(a: LinkAnchor): number {
  return (a.sourceTop + a.sourceBottom) / 2;
}

function corridorY(bounds: LayoutBounds, lane: number): number {
  return Math.max(8 + lane, bounds.linkBandBottom - 12 - lane);
}

function gutterY(bounds: LayoutBounds, lane: number): number {
  const mid = (bounds.gutterTop + bounds.gutterBottom) / 2;
  const half = Math.max(4, (bounds.gutterBottom - bounds.gutterTop) / 2 - 2);
  const spread = Math.min(half, lane * 1.25);
  return mid + (lane % 2 === 0 ? spread : -spread);
}

/** Horizontal lane just above worker cards (never through card bodies). */
function workerCorridorY(bounds: LayoutBounds, lane: number): number {
  return Math.max(bounds.gutterBottom + 4 + lane, bounds.workerRowTop - 12 - lane * 2);
}

function routeSameRow(a: LinkAnchor, bounds: LayoutBounds, lane: number): [number, number][] {
  const isWorker = a.sourceRow === "worker";
  const sy = sourceMidY(a);
  const enterY = a.targetTop;

  if (a.ex >= a.sx) {
    const yC = isWorker ? workerCorridorY(bounds, lane) : corridorY(bounds, lane);
    return [
      [a.sx, sy],
      [a.sx + STUB, sy],
      [a.sx + STUB, yC],
      [a.tx, yC],
      [a.tx, enterY],
    ];
  }

  const yLow = isWorker ? bounds.workerRowBottom - 8 - lane : gutterY(bounds, lane);
  return [
    [a.sx, sy],
    [a.sx + STUB, sy],
    [a.sx + STUB, yLow],
    [a.tx, yLow],
    [a.tx, enterY],
  ];
}

function routeControlToWorker(a: LinkAnchor, bounds: LayoutBounds, lane: number): [number, number][] {
  const yG = gutterY(bounds, lane);
  return [
    [a.scx, a.sourceBottom],
    [a.scx, yG],
    [a.tx, yG],
    [a.tx, a.targetTop],
  ];
}

function routeWorkerToControl(a: LinkAnchor, bounds: LayoutBounds, lane: number): [number, number][] {
  const yG = gutterY(bounds, lane);
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
