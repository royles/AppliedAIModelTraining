import { rayStatusToFields } from "./rayStatusFields";

type Props = {
  status: Record<string, unknown> | null | undefined;
  loading?: boolean;
};

export function RayStatusPanel({ status, loading }: Props) {
  const fields = rayStatusToFields(status);

  return (
    <div className="ray-status-panel">
      <h4 className="ray-status-heading">Ray status</h4>
      {loading && fields.length === 0 ? (
        <p className="hint">Loading Ray status…</p>
      ) : fields.length === 0 ? (
        <p className="hint">No Ray status available.</p>
      ) : (
        <ul className="kv ray-status-kv">
          {fields.map((row) => (
            <li key={row.id} className={row.tone === "error" ? "task-error" : undefined}>
              <span className="ray-status-label">{row.label}</span>
              <span className="ray-status-value">
                {row.id === "initialized" ? (
                  <span className={`pill ray-pill ${row.value === "Yes" ? "succeeded" : "provisioning"}`}>
                    {row.value}
                  </span>
                ) : row.id === "job_ray_address" || row.id === "runtime" ? (
                  <code>{row.value}</code>
                ) : (
                  row.value
                )}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
