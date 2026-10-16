import type { TrainingJob } from "./api";
import { jobArchitectureSummary, jobInitMode } from "./jobUtils";
import { confirmDeleteTask } from "./taskDelete";

type Props = {
  jobs: TrainingJob[];
  onSelect: (jobId: number) => void;
  onDelete: (jobId: number) => void | Promise<void>;
};

export function TaskCardGrid({ jobs, onSelect, onDelete }: Props) {
  if (jobs.length === 0) {
    return (
      <div className="empty-state">
        <p>No model tasks yet.</p>
        <p className="hint">Use <strong>Model Tasks</strong> in the sidebar to configure and start training.</p>
      </div>
    );
  }

  return (
    <div className="task-card-grid">
      {jobs.map((job) => (
        <article key={job.id} className="task-card">
          <button
            type="button"
            className="task-card-main"
            onClick={() => onSelect(job.id)}
          >
            <div className="task-card-head">
              <span className="task-card-id">#{job.id}</span>
              <span className={`pill ${job.status}`}>{job.status}</span>
            </div>
            <h3 className="task-card-title">{job.name}</h3>
            <ul className="task-card-kv">
              <li><span>Mode</span><span>{jobInitMode(job)}</span></li>
              <li><span>Architecture</span><span>{jobArchitectureSummary(job)}</span></li>
              <li><span>Workers</span><span>{job.num_workers} · GPU/worker {job.gpus_per_worker}</span></li>
              <li><span>Steps</span><span>{job.max_steps}</span></li>
              <li><span>Batch</span><span>{job.batch_size}</span></li>
              <li><span>LR</span><span>{job.learning_rate}</span></li>
              <li><span>Ray</span><span><code>{job.ray_address ?? "—"}</code></span></li>
            </ul>
            <span className="task-card-cta">View logs →</span>
          </button>
          <button
            type="button"
            className="btn-danger task-card-delete"
            onClick={() => {
              if (confirmDeleteTask(job)) {
                void onDelete(job.id);
              }
            }}
          >
            Delete
          </button>
        </article>
      ))}
    </div>
  );
}
