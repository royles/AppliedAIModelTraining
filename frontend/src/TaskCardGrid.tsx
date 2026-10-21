import { useMemo } from "react";
import type { Project, TrainingJob } from "./api";
import { TaskCard } from "./TaskCard";
import { canRunJob, canStopJob } from "./jobActions";

type Props = {
  projects: Project[];
  jobs: TrainingJob[];
  onSelect: (jobId: number) => void;
  onRequestDelete: (jobId: number) => void;
  onRun: (jobId: number) => void;
  onStop: (jobId: number) => void;
  jobBusy?: boolean;
};

type ProjectJobGroup = {
  projectId: number;
  projectName: string;
  jobs: TrainingJob[];
};

function groupJobsByProject(projects: Project[], jobs: TrainingJob[]): ProjectJobGroup[] {
  const byProject = new Map<number, TrainingJob[]>();
  for (const job of jobs) {
    const list = byProject.get(job.experiment_id) ?? [];
    list.push(job);
    byProject.set(job.experiment_id, list);
  }

  const groups: ProjectJobGroup[] = [];
  const sortedProjects = [...projects].sort((a, b) => a.name.localeCompare(b.name));

  for (const project of sortedProjects) {
    const projectJobs = byProject.get(project.id);
    if (!projectJobs?.length) continue;
    groups.push({
      projectId: project.id,
      projectName: project.name,
      jobs: [...projectJobs].sort((a, b) => b.id - a.id),
    });
    byProject.delete(project.id);
  }

  for (const [projectId, projectJobs] of byProject) {
    groups.push({
      projectId,
      projectName: `Project #${projectId}`,
      jobs: [...projectJobs].sort((a, b) => b.id - a.id),
    });
  }

  return groups;
}

function JobRunCard({
  job,
  jobBusy,
  onSelect,
  onRun,
  onStop,
  onRequestDelete,
}: {
  job: TrainingJob;
  jobBusy?: boolean;
  onSelect: () => void;
  onRun: () => void;
  onStop: () => void;
  onRequestDelete: () => void;
}) {
  return (
    <div className="task-card-wrap">
      <TaskCard job={job} onSelect={onSelect} />
      <div className="task-card-actions">
        <button
          type="button"
          className="btn-secondary btn-compact task-card-action-run"
          disabled={jobBusy || !canRunJob(job.status)}
          onClick={onRun}
        >
          Run
        </button>
        <button
          type="button"
          className="btn-secondary btn-compact"
          disabled={jobBusy || !canStopJob(job.status)}
          onClick={onStop}
        >
          Stop
        </button>
        <button
          type="button"
          className="btn-danger btn-compact"
          disabled={jobBusy}
          onClick={onRequestDelete}
        >
          Delete
        </button>
      </div>
    </div>
  );
}

export function TaskCardGrid({
  projects,
  jobs,
  onSelect,
  onRequestDelete,
  onRun,
  onStop,
  jobBusy,
}: Props) {
  const groups = useMemo(() => groupJobsByProject(projects, jobs), [projects, jobs]);

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
    <div className="task-runs-by-project">
      {groups.map((group) => (
        <section
          key={group.projectId}
          className="card task-runs-project-panel"
          aria-labelledby={`task-runs-project-${group.projectId}`}
        >
          <header className="task-runs-project-header">
            <h3 id={`task-runs-project-${group.projectId}`}>{group.projectName}</h3>
            <span className="hint">
              {group.jobs.length} run{group.jobs.length === 1 ? "" : "s"}
            </span>
          </header>
          <div className="task-card-grid">
            {group.jobs.map((job) => (
              <JobRunCard
                key={job.id}
                job={job}
                jobBusy={jobBusy}
                onSelect={() => onSelect(job.id)}
                onRun={() => onRun(job.id)}
                onStop={() => onStop(job.id)}
                onRequestDelete={() => onRequestDelete(job.id)}
              />
            ))}
          </div>
        </section>
      ))}
    </div>
  );
}
