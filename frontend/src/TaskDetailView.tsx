import { useCallback, useEffect, useRef, useState } from "react";
import { api, type JobLogs, type TrainingJob } from "./api";
import { jobArchitectureSummary, jobInitMode } from "./jobUtils";
import { ModelInferenceLinks } from "./ModelInferenceLinks";
import { confirmDeleteTask } from "./taskDelete";

type Props = {
  job: TrainingJob;
  onBack: () => void;
  onDelete: (jobId: number) => void | Promise<void>;
};

function formatLogLines(lines: { ts: string; message: string }[]): string {
  if (lines.length === 0) return "No entries yet.";
  return lines.map((l) => `[${l.ts}] ${l.message}`).join("\n");
}

function scrollLogToBottom(el: HTMLDivElement | null) {
  if (el) {
    el.scrollTop = el.scrollHeight;
  }
}

export function TaskDetailView({ job, onBack, onDelete }: Props) {
  const [logs, setLogs] = useState<JobLogs | null>(null);
  const [error, setError] = useState<string | null>(null);
  const rayStatusLogRef = useRef<HTMLDivElement>(null);
  const frameworkLogRef = useRef<HTMLDivElement>(null);
  const trainingLogRef = useRef<HTMLDivElement>(null);

  const loadLogs = useCallback(async () => {
    try {
      const data = await api.getJobLogs(job.id);
      setLogs(data);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [job.id]);

  useEffect(() => {
    loadLogs();
    const id = window.setInterval(loadLogs, 3000);
    return () => window.clearInterval(id);
  }, [loadLogs]);

  const rayStatusText = logs
    ? JSON.stringify(logs.ray_status, null, 2)
    : "Loading Ray status…";

  useEffect(() => {
    scrollLogToBottom(rayStatusLogRef.current);
    scrollLogToBottom(frameworkLogRef.current);
    scrollLogToBottom(trainingLogRef.current);
  }, [logs, rayStatusText]);

  return (
    <div className="task-detail">
      <div className="task-detail-toolbar">
        <button type="button" className="btn-secondary" onClick={onBack}>
          ← All tasks
        </button>
        <div className="task-detail-title-wrap">
          <h2>{job.name}</h2>
          <span className={`pill ${job.status}`}>{job.status}</span>
        </div>
        <button
          type="button"
          className="btn-danger task-detail-delete"
          onClick={() => {
            if (confirmDeleteTask(job)) {
              void onDelete(job.id);
            }
          }}
        >
          Delete task
        </button>
      </div>

      <article className="card task-detail-summary">
        <ul className="kv">
          <li><span>Task id</span><span>#{job.id}</span></li>
          <li><span>Init</span><span>{jobInitMode(job)}</span></li>
          <li><span>Architecture</span><span>{jobArchitectureSummary(job)}</span></li>
          <li><span>Workers / GPU</span><span>{job.num_workers} / {job.gpus_per_worker}</span></li>
          <li><span>Max steps</span><span>{job.max_steps}</span></li>
          <li><span>Ray</span><span><code>{job.ray_address ?? "local"}</code></span></li>
        </ul>
      </article>

      <ModelInferenceLinks job={job} />

      {error && <div className="banner error">{error}</div>}

      <div className="log-split">
        <section className="card log-panel">
          <h3>Framework &amp; Ray status</h3>
          <p className="hint">Provisioning, workers, Ray cluster snapshot (refreshes every 3s)</p>
          <div className="log-ray-status" ref={rayStatusLogRef}>
            <strong>Ray status</strong>
            <pre>{rayStatusText}</pre>
          </div>
          <div className="log-stream" ref={frameworkLogRef}>
            <pre>{formatLogLines(logs?.framework ?? [])}</pre>
          </div>
        </section>
        <section className="card log-panel">
          <h3>Training steps</h3>
          <p className="hint">Loss, learning rate, and checkpoint steps from the train loop</p>
          <div className="log-stream" ref={trainingLogRef}>
            <pre>{formatLogLines(logs?.training ?? [])}</pre>
          </div>
        </section>
      </div>
    </div>
  );
}
