import type { TrainingJob } from "./api";
import { jobArchitectureSummary, jobDatasetSummary, jobInitMode } from "./jobUtils";
import { ModelInferenceLinks } from "./ModelInferenceLinks";

type BodyProps = {
  job: TrainingJob;
  showCta?: boolean;
  onOpenRegistry?: () => void;
};

/** Shared task summary markup (Model Tasks grid + task detail header). */
export function TaskCardBody({ job, showCta = false, onOpenRegistry }: BodyProps) {
  return (
    <>
      <div className="task-card-head">
        <span className="task-card-id">#{job.id}</span>
        <span className={`pill ${job.status}`}>{job.status}</span>
      </div>
      <h3 className="task-card-title">{job.name}</h3>
      <ul className="task-card-kv">
        <li><span>Mode</span><span>{jobInitMode(job)}</span></li>
        <li><span>Dataset</span><span>{jobDatasetSummary(job)}</span></li>
        <li><span>Architecture</span><span>{jobArchitectureSummary(job)}</span></li>
        <li><span>Workers</span><span>{job.num_workers} · GPU/worker {job.gpus_per_worker}</span></li>
        <li><span>Steps</span><span>{job.max_steps}</span></li>
        <li><span>Batch</span><span>{job.batch_size}</span></li>
        <li><span>LR</span><span>{job.learning_rate}</span></li>
        <li><span>Ray</span><span><code>{job.ray_address ?? "—"}</code></span></li>
      </ul>
      {job.error_message && (
        <p className="task-card-error" title={job.error_message}>{job.error_message}</p>
      )}
      {showCta && <span className="task-card-cta">View logs →</span>}
      <ModelInferenceLinks job={job} compact onOpenRegistry={onOpenRegistry} />
    </>
  );
}

type CardProps = BodyProps & {
  variant?: "grid" | "detail";
  onSelect?: () => void;
};

export function TaskCard({ job, variant = "grid", showCta, onSelect, onOpenRegistry }: CardProps) {
  const isDetail = variant === "detail";
  return (
    <article className={`task-card ${isDetail ? "task-card-detail task-card-selected" : ""}`}>
      {isDetail ? (
        <div className="task-card-main task-card-main-static">
          <TaskCardBody job={job} onOpenRegistry={onOpenRegistry} />
        </div>
      ) : (
        <button type="button" className="task-card-main" onClick={onSelect}>
          <TaskCardBody job={job} showCta={showCta ?? true} onOpenRegistry={onOpenRegistry} />
        </button>
      )}
    </article>
  );
}
