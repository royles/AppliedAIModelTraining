import { useLayoutEffect, useMemo, useRef, useState } from "react";
import type { RayClusterComponent, RayClusterLink } from "./rayClusterTypes";
import { parseDistributed, parseTopology } from "./rayClusterTypes";

type Props = {
  status: Record<string, unknown> | null | undefined;
  jobStatus: string;
  loading?: boolean;
  cdswEngines?: number;
};

const ROLE_ORDER: Record<string, number> = {
  driver: 0,
  master: 1,
  controller: 2,
  train_worker: 3,
  worker: 4,
  worker_pending: 5,
  worker_dead: 6,
};

const EMPTY_LINKS: RayClusterLink[] = [];

const ROLE_TIER: Record<string, string> = {
  driver: "tier-driver",
  master: "tier-head",
  controller: "tier-controller",
  train_worker: "tier-train",
  worker: "tier-workers",
  worker_pending: "tier-workers",
  worker_dead: "tier-workers",
};

function tierForRole(role: string): string {
  return ROLE_TIER[role] ?? "tier-workers";
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

function NodeCard({ node }: { node: RayClusterComponent }) {
  const roleClass = node.role.replace(/_/g, "-");
  return (
    <article
      className={`ray-node-card ${node.healthy ? "healthy" : "unhealthy"} role-${roleClass}`}
      data-node-id={node.id}
    >
      <header className="ray-node-card-head">
        <span className="ray-node-role">{node.role.replace(/_/g, " ")}</span>
        <span className={`ray-node-health ${node.healthy ? "up" : "down"}`}>
          {node.healthy ? "●" : "○"}
        </span>
      </header>
      <h6 className="ray-node-title">{node.title}</h6>
      <p className="ray-node-sub">{node.subtitle}</p>
      {node.host ? <code className="ray-node-host">{node.host}</code> : null}
      {node.metrics.length > 0 && (
        <ul className="ray-node-metrics">
          {node.metrics.slice(0, 6).map((m) => (
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

type Line = { x1: number; y1: number; x2: number; y2: number; healthy: boolean; key: string };

function linesEqual(a: Line[], b: Line[]): boolean {
  if (a.length !== b.length) {
    return false;
  }
  for (let i = 0; i < a.length; i += 1) {
    const x = a[i];
    const y = b[i];
    if (
      x.key !== y.key ||
      x.healthy !== y.healthy ||
      Math.abs(x.x1 - y.x1) > 0.5 ||
      Math.abs(x.y1 - y.y1) > 0.5 ||
      Math.abs(x.x2 - y.x2) > 0.5 ||
      Math.abs(x.y2 - y.y2) > 0.5
    ) {
      return false;
    }
  }
  return true;
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
    (distributed as { mode_label?: string }).mode_label ??
    distributed.summary;

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
        key: "backend",
        label: "Backend",
        value: 0,
        display: distributed.backend ?? "—",
        unit: "text",
      },
      {
        key: "ranks",
        label: "Ranks live",
        value: obs.ranks_live,
        display: `${obs.ranks_live}/${obs.world_size || cfg.ray_train_workers || "?"}`,
        unit: "count",
      },
    ];
    if (distributed.collective_backend) {
      extra.push({
        key: "collective",
        label: "Collective",
        value: 0,
        display: String(distributed.collective_backend),
        unit: "text",
      });
    }
    const keys = new Set(c.metrics.map((m) => m.key));
    const merged = [...c.metrics, ...extra.filter((e) => !keys.has(e.key))];
    return { ...c, metrics: merged };
  });
}

export function RayClusterMap({ status, loading, cdswEngines }: Props) {
  const topology = useMemo(() => parseTopology(status), [status]);
  const distributed = useMemo(() => parseDistributed(status), [status]);
  const containerRef = useRef<HTMLDivElement>(null);
  const [lines, setLines] = useState<Line[]>([]);

  const components = useMemo(
    () => enrichComponents(topology?.components ?? [], distributed),
    [topology, distributed],
  );
  const links = topology?.links ?? EMPTY_LINKS;
  const summary = topology?.summary;
  const alerts = clusterAlerts(status, topology?.note);

  const tiers = useMemo(() => {
    const grouped = new Map<string, RayClusterComponent[]>();
    for (const c of components) {
      const tier = tierForRole(c.role);
      const list = grouped.get(tier) ?? [];
      list.push(c);
      grouped.set(tier, list);
    }
    for (const [, list] of grouped) {
      list.sort((a, b) => (ROLE_ORDER[a.role] ?? 9) - (ROLE_ORDER[b.role] ?? 9));
    }
    return grouped;
  }, [components]);

  useLayoutEffect(() => {
    const root = containerRef.current;
    if (!root || links.length === 0) {
      setLines((prev) => (prev.length === 0 ? prev : []));
      return;
    }
    const measure = () => {
      const rootBox = root.getBoundingClientRect();
      const next: Line[] = [];
      for (const link of links) {
        const fromEl = root.querySelector(`[data-node-id="${link.from}"]`);
        const toEl = root.querySelector(`[data-node-id="${link.to}"]`);
        if (!fromEl || !toEl) continue;
        const a = fromEl.getBoundingClientRect();
        const b = toEl.getBoundingClientRect();
        next.push({
          x1: a.right - rootBox.left,
          y1: a.top + a.height / 2 - rootBox.top,
          x2: b.left - rootBox.left,
          y2: b.top + b.height / 2 - rootBox.top,
          healthy: link.healthy,
          key: `${link.from}-${link.to}-${link.kind}`,
        });
      }
      setLines((prev) => (linesEqual(prev, next) ? prev : next));
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(root);
    return () => ro.disconnect();
  }, [links, components, loading]);

  if (loading && !topology) {
    return <p className="hint">Loading cluster map…</p>;
  }

  const tierOrder = ["tier-driver", "tier-head", "tier-controller", "tier-train", "tier-workers"];

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

      <div className="ray-cluster-canvas ray-cluster-canvas-ltr">
        <svg className="ray-cluster-svg" aria-hidden>
          {lines.map((ln) => (
            <line
              key={ln.key}
              x1={ln.x1}
              y1={ln.y1}
              x2={ln.x2}
              y2={ln.y2}
              className={ln.healthy ? "link-ok" : "link-bad"}
            />
          ))}
        </svg>

        {tierOrder.map((tier) => {
          const nodes = tiers.get(tier);
          if (!nodes?.length) return null;
          return (
            <div key={tier} className={`ray-cluster-tier ${tier}`}>
              {nodes.map((node) => (
                <NodeCard key={node.id} node={node} />
              ))}
            </div>
          );
        })}
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
        {distributed?.summary && <span className="ray-cluster-legend-summary">{distributed.summary}</span>}
      </footer>
    </div>
  );
}
