import type { TrainingJob } from "./api";
import { TaskCard } from "./TaskCard";
import { canRunJob, canStopJob } from "./jobActions";

type Props = {
  jobs: TrainingJob[];
  onSelect: (jobId: number) => void;
  onRequestDelete: (jobId: number) => void;
  onRun: (jobId: number) => void;
  onStop: (jobId: number) => void;
  jobBusy?: boolean;
};

export function TaskCardGrid({ jobs, onSelect, onRequestDelete, onRun, onStop, jobBusy }: Props) {
  if (jobs.length === 0) {
    return (
      <div className="empty-state">
        <p>No model tasks yet.</p>
        <p className="hint">
          Use <strong>Model Tasks</strong> in the sidebar to configure and start training.
        </p>
      </div>
    );
  }

  return (
    <div className="task-card-grid">
      {jobs.map((job) => (
        <div key={job.id} className="task-card-wrap">
          <TaskCard job={job} onSelect={() => onSelect(job.id)} />
          <div className="task-card-actions">
            <button
              type="button"
              className="btn-compact"
              disabled={jobBusy || !canRunJob(job.status)}
              onClick={() => onRun(job.id)}
            >
              Run
            </button>
            <button
              type="button"
              className="btn-secondary btn-compact"
              disabled={jobBusy || !canStopJob(job.status)}
              onClick={() => onStop(job.id)}
            >
              Stop
            </button>
            <button
              type="button"
              className="btn-danger btn-compact"
              disabled={jobBusy}
              onClick={() => onRequestDelete(job.id)}
            >
              Delete
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}
