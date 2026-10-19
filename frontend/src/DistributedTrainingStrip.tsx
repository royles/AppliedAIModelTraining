import type { DistributedTrainingStatus } from "./rayClusterTypes";

type Props = {
  distributed: DistributedTrainingStatus | null;
  jobStatus: string;
};

function StatusPill({ distributed, jobStatus }: Props) {
  if (!distributed) {
    return <span className="dist-pill dist-pill-muted">Distributed status pending</span>;
  }
  if (distributed.status === "distributed_active") {
    return <span className="dist-pill dist-pill-live">Distributed training live</span>;
  }
  if (distributed.status === "partial") {
    return <span className="dist-pill dist-pill-warn">Workers joining</span>;
  }
  if (distributed.status === "completed" || jobStatus === "succeeded") {
    return <span className="dist-pill dist-pill-ok">Training completed</span>;
  }
  if (distributed.status === "starting" || jobStatus === "running" || jobStatus === "provisioning") {
    return <span className="dist-pill dist-pill-warn">Cluster starting</span>;
  }
  return <span className="dist-pill dist-pill-muted">Not started</span>;
}

export function DistributedTrainingStrip({ distributed, jobStatus }: Props) {
  const cfg = distributed?.configured;
  const obs = distributed?.observed;

  return (
    <section className="distributed-strip" aria-label="Distributed training">
      <div className="distributed-strip-header">
        <div>
          <h5 className="distributed-strip-title">Model training on Ray Train</h5>
          <p className="distributed-strip-summary">
            {distributed?.summary ??
              "TorchTrainer runs the model on Ray worker processes across the cluster."}
          </p>
        </div>
        <StatusPill distributed={distributed} jobStatus={jobStatus} />
      </div>

      <div className="distributed-indicators">
        <Indicator
          ok={Boolean(distributed?.backend)}
          label="Backend"
          value={distributed?.backend ?? "—"}
        />
        <Indicator
          ok={Boolean(distributed?.multi_worker)}
          label="Mode"
          value={
            distributed?.multi_worker
              ? `Distributed (${obs?.world_size ?? cfg?.ray_train_workers ?? "?"} ranks)`
              : "Single worker"
          }
        />
        <Indicator
          ok={Boolean(distributed?.active)}
          label="Train loop"
          value={
            distributed?.active
              ? `${obs?.ranks_live ?? 0}/${obs?.world_size || cfg?.ray_train_workers || "?"} ranks active`
              : "Waiting for ranks"
          }
        />
        <Indicator
          ok={Boolean(distributed?.observed?.all_ranks_reported)}
          label="All ranks"
          value={distributed?.observed?.all_ranks_reported ? "Reported" : "Pending"}
        />
        <Indicator
          ok={(cfg?.cdsw_worker_engines ?? 0) > 0}
          label="CDSW engines"
          value={String(cfg?.cdsw_worker_engines ?? 0)}
        />
        <Indicator
          ok={(cfg?.gpus_per_worker ?? 0) > 0}
          label="GPU / worker"
          value={String(cfg?.gpus_per_worker ?? 0)}
        />
      </div>

      {distributed && distributed.ranks.length > 0 && (
        <ul className="distributed-ranks">
          {distributed.ranks.map((r) => (
            <li key={r.rank} className="distributed-rank-card">
              <span className="distributed-rank-badge">Rank {r.display_rank}</span>
              <span className="distributed-rank-device">{r.device}</span>
              {r.host ? <span className="distributed-rank-host">{r.host}</span> : null}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

function Indicator({ ok, label, value }: { ok: boolean; label: string; value: string }) {
  return (
    <div className={`distributed-indicator ${ok ? "ok" : ""}`}>
      <span className={`distributed-indicator-dot ${ok ? "on" : ""}`} aria-hidden />
      <span className="distributed-indicator-label">{label}</span>
      <span className="distributed-indicator-value">{value}</span>
    </div>
  );
}
