import { useLayoutEffect, useMemo, useRef, useState } from "react";
import type { RayClusterComponent, RayClusterLink } from "./rayClusterTypes";
import {
  routeClusterLinks,
  routedLinksEqual,
  type LayoutBounds,
  type NodeRow,
  type RoutedLink,
} from "./rayClusterRouting";
import { parseDistributed, parseTopology } from "./rayClusterTypes";

type Props = {
  status: Record<string, unknown> | null | undefined;
  jobStatus: string;
  loading?: boolean;
  cdswEngines?: number;
};

const CONTROL_ORDER: Record<string, number> = {
  driver: 0,
  master: 1,
  controller: 2,
  train_worker: 3,
};

const WORKER_ORDER: Record<string, number> = {
  worker: 0,
  worker_pending: 1,
  worker_dead: 2,
};

const EMPTY_LINKS: RayClusterLink[] = [];

function nodeRow(role: string): NodeRow {
  if (role === "worker" || role === "worker_pending" || role === "worker_dead") {
    return "worker";
  }
  return "control";
}

function clusterAlerts(
  status: Record<string, unknown> | null | undefined,
  topologyNote?: string,
): string[] {
  const out: string[] = [];
  for (const key of ["cluster_error", "note"]) {
    const v = status?.[key];
    if (typeof v === "string" && v.trim()) {
      out.push(v.trim());
    }
  }
  if (topologyNote?.trim()) {
    out.push(topologyNote.trim());
  }
  return out;
}

function NodeCard({ node, row }: { node: RayClusterComponent; row: NodeRow }) {
  const roleClass = node.role.replace(/_/g, "-");
  const compact = row === "worker";
  return (
    <article
      className={`ray-node-card ${node.healthy ? "healthy" : "unhealthy"} role-${roleClass}${
        compact ? " ray-node-card-compact" : ""
      }`}
      data-node-id={node.id}
      data-node-row={row}
    >
      <header className="ray-node-card-head">
        <span className="ray-node-role">{node.role.replace(/_/g, " ")}</span>
        <span className={`ray-node-health ${node.healthy ? "up" : "down"}`}>
          {node.healthy ? "●" : "○"}
        </span>
      </header>
      <h6 className="ray-node-title">{node.title}</h6>
      {!compact && <p className="ray-node-sub">{node.subtitle}</p>}
      {node.host ? <code className="ray-node-host">{node.host}</code> : null}
      {node.metrics.length > 0 && (
        <ul className="ray-node-metrics">
          {node.metrics.slice(0, compact ? 3 : 4).map((m) => (
            <li key={m.key}>
              <span>{m.label}</span>
              <strong>{m.display}</strong>
            </li>
          ))}
        </ul>
      )}
    </article>
  );
}

function enrichComponents(
  components: RayClusterComponent[],
  distributed: ReturnType<typeof parseDistributed>,
): RayClusterComponent[] {
  if (!distributed) {
    return components;
  }
  const cfg = distributed.configured;
  const obs = distributed.observed;
  const modeLabel =
    (distributed as { mode_label?: string }).mode_label ?? distributed.summary;

  return components.map((c) => {
    if (c.id !== "train-controller") {
      return c;
    }
    const extra = [
      {
        key: "mode",
        label: "Scale",
        value: 0,
        display:
          modeLabel ||
          (cfg.ray_train_workers > 1
            ? `${cfg.ray_train_workers} Train ranks`
            : cfg.cdsw_worker_engines > 1
              ? `${cfg.cdsw_worker_engines} engines · 1 Train rank`
              : "Single Train rank"),
        unit: "text",
      },
      {
        key: "ranks",
        label: "Ranks",
        value: obs.ranks_live,
        display: `${obs.ranks_live}/${obs.world_size || cfg.ray_train_workers || "?"}`,
        unit: "count",
      },
    ];
    const keys = new Set(c.metrics.map((m) => m.key));
    const merged = [...c.metrics, ...extra.filter((e) => !keys.has(e.key))];
    return { ...c, metrics: merged.slice(0, 5) };
  });
}

function sortNodes(
  nodes: RayClusterComponent[],
  order: Record<string, number>,
): RayClusterComponent[] {
  return [...nodes].sort((a, b) => {
    const d = (order[a.role] ?? 9) - (order[b.role] ?? 9);
    if (d !== 0) {
      return d;
    }
    return a.id.localeCompare(b.id);
  });
}

export function RayClusterMap({ status, loading, cdswEngines }: Props) {
  const topology = useMemo(() => parseTopology(status), [status]);
  const distributed = useMemo(() => parseDistributed(status), [status]);
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLDivElement>(null);
  const [routes, setRoutes] = useState<RoutedLink[]>([]);
  const [fit, setFit] = useState({ scale: 1, height: 0 });

  const components = useMemo(
    () => enrichComponents(topology?.components ?? [], distributed),
    [topology, distributed],
  );
  const links = topology?.links ?? EMPTY_LINKS;
  const summary = topology?.summary;
  const alerts = clusterAlerts(status, topology?.note);

  const controlNodes = useMemo(
    () => sortNodes(components.filter((c) => nodeRow(c.role) === "control"), CONTROL_ORDER),
    [components],
  );
  const workerNodes = useMemo(
    () => sortNodes(components.filter((c) => nodeRow(c.role) === "worker"), WORKER_ORDER),
    [components],
  );

  useLayoutEffect(() => {
    const root = containerRef.current;
    const canvasEl = canvasRef.current;
    if (!root || !canvasEl) {
      setRoutes((prev) => (prev.length === 0 ? prev : []));
      return;
    }

    const measure = () => {
      const canvasBox = canvasEl.getBoundingClientRect();
      const viewport = root.querySelector(".ray-cluster-viewport") as HTMLElement | null;
      const pad = 4;

      const linkBand = canvasEl.querySelector(".ray-cluster-link-band");
      const controlRow = canvasEl.querySelector(".ray-cluster-row-control .ray-cluster-row-nodes");
      const workerRow = canvasEl.querySelector(".ray-cluster-row-workers .ray-cluster-row-nodes");

      const linkBandBottom = linkBand
        ? linkBand.getBoundingClientRect().bottom - canvasBox.top
        : 32;

      const controlRect = controlRow?.getBoundingClientRect();
      const workerRect = workerRow?.getBoundingClientRect();
      const gutterEl = canvasEl.querySelector(".ray-cluster-row-gutter");
      const gutterRect = gutterEl?.getBoundingClientRect();

      const controlRowTop = controlRect ? controlRect.top - canvasBox.top : linkBandBottom + 4;
      const controlRowBottom = controlRect ? controlRect.bottom - canvasBox.top : linkBandBottom + 80;
      const workerRowTop = workerRect ? workerRect.top - canvasBox.top : controlRowBottom + 12;
      const workerRowBottom = workerRect ? workerRect.bottom - canvasBox.top : controlRowTop + 160;
      const gutterTop = gutterRect ? gutterRect.top - canvasBox.top : controlRowBottom;
      const gutterBottom = gutterRect ? gutterRect.bottom - canvasBox.top : workerRowTop;

      const bounds: LayoutBounds = {
        linkBandBottom,
        controlRowTop,
        controlRowBottom,
        gutterTop,
        gutterBottom,
        workerRowTop,
        workerRowBottom,
      };

      const anchors = [];
      for (const link of links) {
        const fromEl = root.querySelector(`[data-node-id="${link.from}"]`);
        const toEl = root.querySelector(`[data-node-id="${link.to}"]`);
        if (!fromEl || !toEl) continue;
        const a = fromEl.getBoundingClientRect();
        const b = toEl.getBoundingClientRect();
        const sourceRow = (fromEl.getAttribute("data-node-row") as NodeRow) || "control";
        const targetRow = (toEl.getAttribute("data-node-row") as NodeRow) || "control";
        anchors.push({
          key: `${link.from}-${link.to}-${link.kind}`,
          healthy: link.healthy,
          sourceRow,
          targetRow,
          sx: a.right - canvasBox.left + pad,
          scx: a.left + a.width / 2 - canvasBox.left,
          sourceTop: a.top - canvasBox.top,
          sourceBottom: a.bottom - canvasBox.top,
          ex: b.left - canvasBox.left - pad,
          tx: b.left + b.width / 2 - canvasBox.left,
          targetTop: b.top - canvasBox.top,
          targetBottom: b.bottom - canvasBox.top,
        });
      }

      const next = links.length > 0 ? routeClusterLinks(anchors, bounds) : [];
      setRoutes((prev) => (routedLinksEqual(prev, next) ? prev : next));

      const naturalW = canvasEl.scrollWidth;
      const naturalH = canvasEl.scrollHeight;
      const availW = (viewport?.clientWidth ?? root.clientWidth) - 4;
      const scale =
        naturalW > 0 && availW > 0 && naturalW > availW
          ? Math.max(0.5, Math.min(1, availW / naturalW))
          : 1;
      const height = Math.ceil(naturalH * scale) + 4;
      setFit((prev) => (prev.scale === scale && prev.height === height ? prev : { scale, height }));
    };

    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(root);
    ro.observe(canvasEl);
    return () => ro.disconnect();
  }, [links, components, loading]);

  if (loading && !topology) {
    return <p className="hint">Loading cluster map…</p>;
  }

  if (!topology || components.length === 0) {
    return (
      <div className="ray-cluster-map empty">
        <p className="hint">
          Cluster map unavailable — start Ray and open a running task.
          {cdswEngines && cdswEngines > 0 ? ` (${cdswEngines} CDSW engine(s) configured.)` : ""}
        </p>
      </div>
    );
  }

  return (
    <div className="ray-cluster-map" ref={containerRef}>
      {alerts.length > 0 && (
        <ul className="ray-cluster-alerts">
          {alerts.map((msg, i) => (
            <li key={i}>{msg}</li>
          ))}
        </ul>
      )}

      <div
        className="ray-cluster-viewport"
        style={{ height: fit.height > 0 ? `${fit.height}px` : undefined }}
      >
        <div
          ref={canvasRef}
          className="ray-cluster-canvas ray-cluster-canvas-grid"
          style={{
            transform: fit.scale < 1 ? `scale(${fit.scale})` : undefined,
            transformOrigin: "top left",
          }}
        >
          <svg className="ray-cluster-svg ray-cluster-svg-overlay" aria-hidden>
            {routes.map((ln) => (
              <path
                key={ln.key}
                d={ln.d}
                fill="none"
                className={ln.healthy ? "link-ok" : "link-bad"}
              />
            ))}
          </svg>

          <div className="ray-cluster-link-band" aria-hidden />

          <div className="ray-cluster-stage">
            <section className="ray-cluster-row ray-cluster-row-control">
              <header className="ray-cluster-row-label">Control &amp; train</header>
              <div className="ray-cluster-row-nodes">
                {controlNodes.map((node) => (
                  <NodeCard key={node.id} node={node} row="control" />
                ))}
              </div>
            </section>

            <div className="ray-cluster-row-gutter" aria-hidden />

            <section className="ray-cluster-row ray-cluster-row-workers">
              <header className="ray-cluster-row-label">Ray workers</header>
              <div className="ray-cluster-row-nodes">
                {workerNodes.length === 0 ? (
                  <p className="hint ray-cluster-row-empty">No worker nodes in cluster yet.</p>
                ) : (
                  workerNodes.map((node) => (
                    <NodeCard key={node.id} node={node} row="worker" />
                  ))
                )}
              </div>
            </section>
          </div>
        </div>
      </div>

      <footer className="ray-cluster-legend">
        <span><i className="legend-line ok" /> Healthy</span>
        <span><i className="legend-line bad" /> Issue</span>
        <span>GCS {topology.gcs_reachable ? "up" : "down"}</span>
        {summary && (
          <>
            <span>Nodes {summary.alive_nodes ?? 0} alive</span>
            <span>
              CPU {summary.used_cpu ?? 0}/{summary.total_cpu ?? 0} · GPU{" "}
              {summary.used_gpu ?? 0}/{summary.total_gpu ?? 0}
            </span>
          </>
        )}
        {distributed?.summary && (
          <span className="ray-cluster-legend-summary">{distributed.summary}</span>
        )}
      </footer>
    </div>
  );
}
