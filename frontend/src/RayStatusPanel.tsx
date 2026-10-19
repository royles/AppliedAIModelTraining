import {
  rayResourceBarMax,
  rayStatusToDashboard,
  type RayResourceMetric,
  type RayStatusStat,
} from "./rayStatusFields";

type Props = {
  status: Record<string, unknown> | null | undefined;
  loading?: boolean;
  jobWorkers?: number;
};

function MetricBar({
  metric,
  max,
}: {
  metric: RayResourceMetric;
  max: number;
}) {
  const pct = Math.min(100, Math.round((metric.value / max) * 100));
  return (
    <div className="ray-metric-bar" data-accent={metric.accent}>
      <div className="ray-metric-bar-header">
        <span className="ray-metric-bar-label">{metric.label}</span>
        <span className="ray-metric-bar-value">{metric.display}</span>
      </div>
      <div className="ray-metric-bar-track" role="presentation">
        <div
          className="ray-metric-bar-fill"
          style={{ width: `${pct}%` }}
          title={`${metric.display} (${pct}% of peak in this snapshot)`}
        />
      </div>
    </div>
  );
}

function StatCard({ stat, barMax }: { stat: RayStatusStat; barMax: number }) {
  const showBar = stat.numeric !== undefined && stat.id === "nodes";

  return (
    <article className="ray-stat-card">
      <span className="ray-stat-label">{stat.label}</span>
      {stat.format === "pill" ? (
        <span className={`pill ray-pill ${stat.value === "Yes" ? "succeeded" : "provisioning"}`}>
          {stat.value}
        </span>
      ) : stat.format === "code" ? (
        <code className="ray-stat-code">{stat.value}</code>
      ) : (
        <span className="ray-stat-value">{stat.value}</span>
      )}
      {showBar && stat.numeric !== undefined && (
        <div className="ray-metric-bar-track ray-stat-mini-track" role="presentation">
          <div
            className="ray-metric-bar-fill accent-node"
            style={{ width: `${Math.min(100, Math.round((stat.numeric / barMax) * 100))}%` }}
          />
        </div>
      )}
    </article>
  );
}

export function RayStatusPanel({ status, loading, jobWorkers }: Props) {
  const dashboard = rayStatusToDashboard(status);
  const barMax = rayResourceBarMax(dashboard, jobWorkers);
  const { stats, resources, messages } = dashboard;
  const hasContent = stats.length > 0 || resources.length > 0 || messages.length > 0;

  const primaryStats = stats.filter((s) => s.id !== "job_ray_address");
  const addressStat = stats.find((s) => s.id === "job_ray_address");

  return (
    <div className="ray-status-panel ray-dashboard">
      <h4 className="ray-status-heading">Ray cluster</h4>
      {loading && !hasContent ? (
        <p className="hint">Loading Ray status…</p>
      ) : !hasContent ? (
        <p className="hint">No Ray status available.</p>
      ) : (
        <>
          {messages.length > 0 && (
            <div className="ray-dashboard-messages">
              {messages.map((m) => (
                <div key={m.id} className={`ray-dashboard-message tone-${m.tone ?? "default"}`}>
                  <strong>{m.label}</strong>
                  <p>{m.value}</p>
                </div>
              ))}
            </div>
          )}

          {primaryStats.length > 0 && (
            <div className="ray-stat-grid">
              {primaryStats.map((stat) => (
                <StatCard key={stat.id} stat={stat} barMax={barMax} />
              ))}
            </div>
          )}

          {resources.length > 0 && (
            <section className="ray-resource-section" aria-label="Cluster resources">
              <h5 className="ray-resource-heading">Resources</h5>
              <div className="ray-metric-bars">
                {resources.map((metric) => (
                  <MetricBar key={metric.id} metric={metric} max={barMax} />
                ))}
              </div>
            </section>
          )}

          {addressStat && (
            <div className="ray-dashboard-detail">
              <span className="ray-stat-label">{addressStat.label}</span>
              <code>{addressStat.value}</code>
            </div>
          )}
        </>
      )}
    </div>
  );
}
