import type { TrainingJob } from "./api";
import { TaskCard } from "./TaskCard";
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
        <div key={job.id} className="task-card-wrap">
          <TaskCard job={job} onSelect={() => onSelect(job.id)} />
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
        </div>
      ))}
    </div>
  );
}
