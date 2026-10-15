import { useState } from "react";
import type { TrainingJob } from "./api";

function formatLogs(job: TrainingJob): string {
  const parts: string[] = [];
  if (job.error_message) {
    parts.push(`Error:\n${job.error_message}`);
  }
  if (job.metrics_json) {
    try {
      const parsed = JSON.parse(job.metrics_json);
      parts.push(`Metrics:\n${JSON.stringify(parsed, null, 2)}`);
    } catch {
      parts.push(`Metrics:\n${job.metrics_json}`);
    }
  }
  if (parts.length === 0) {
    return "No logs or metrics yet.";
  }
  return parts.join("\n\n");
}

function initMode(job: TrainingJob): string {
  try {
    if (job.config_json) {
      const c = JSON.parse(job.config_json);
      return c.init_mode ?? "—";
    }
  } catch {
    /* ignore */
  }
  return "—";
}

export function JobRunList({ jobs }: { jobs: TrainingJob[] }) {
  const [expandedIds, setExpandedIds] = useState<Set<number>>(() => new Set());

  function toggle(id: number) {
    setExpandedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  }

  if (jobs.length === 0) {
    return <p className="hint">No training runs yet.</p>;
  }

  return (
    <ul className="job-list">
      {jobs.map((job) => {
        const expanded = expandedIds.has(job.id);
        const hasLogs = Boolean(job.metrics_json || job.error_message);
        return (
          <li key={job.id} className="job-run">
            <button
              type="button"
              className="job-run-header"
              onClick={() => toggle(job.id)}
              aria-expanded={expanded}
            >
              <span className="job-chevron" aria-hidden>{expanded ? "▾" : "▸"}</span>
              <span className="job-run-id">#{job.id}</span>
              <span className="job-run-name">{job.name}</span>
              <span className={`pill ${job.status}`}>{job.status}</span>
              <span className="job-run-meta">{initMode(job)}</span>
              <span className="job-run-meta job-run-ray">
                <code>{job.ray_address ?? "local"}</code>
              </span>
              {hasLogs && !expanded && (
                <span className="job-run-log-hint">logs hidden</span>
              )}
            </button>
            {expanded && (
              <div className="job-run-logs">
                <pre>{formatLogs(job)}</pre>
              </div>
            )}
          </li>
        );
      })}
    </ul>
  );
}
