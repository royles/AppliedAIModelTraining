import { useEffect, useState } from "react";
import type { Project, TaskDefinition, TaskVersion, TrainingJob } from "./api";
import { canRunJob, canStopJob } from "./jobActions";

type Props = {
  projects: Project[];
  jobs: TrainingJob[];
  selectedProjectId: number | null;
  onSelectedProjectIdChange: (id: number) => void;
  newProjectName: string;
  onNewProjectNameChange: (v: string) => void;
  newProjectDescription: string;
  onNewProjectDescriptionChange: (v: string) => void;
  onCreateProject: () => void;
  onUpdateProject: (projectId: number, name: string) => void;
  onDeleteProject: (projectId: number) => void;
  onUpdateJobName: (jobId: number, name: string) => void;
  onRequestDeleteJob: (jobId: number) => void;
  onRunJob: (jobId: number) => void;
  onStopJob: (jobId: number) => void;
  projectBusy: boolean;
  jobBusy: boolean;
  onInvalidateBlueprint: () => void;
  taskDefinitions?: TaskDefinition[];
  taskVersions?: TaskVersion[];
  selectedTaskId?: number | null;
  onSelectedTaskIdChange?: (id: number | null) => void;
  onApplyTaskVersion?: (spec: Record<string, unknown>) => void;
};

export function ProjectJobsHierarchy(props: Props) {
  const {
    projects,
    jobs,
    selectedProjectId,
    onSelectedProjectIdChange,
    newProjectName,
    onNewProjectNameChange,
    newProjectDescription,
    onNewProjectDescriptionChange,
    onCreateProject,
    onUpdateProject,
    onDeleteProject,
    onUpdateJobName,
    onRequestDeleteJob,
    onRunJob,
    onStopJob,
    projectBusy,
    jobBusy,
    onInvalidateBlueprint,
    taskDefinitions = [],
    taskVersions = [],
    selectedTaskId = null,
    onSelectedTaskIdChange,
    onApplyTaskVersion,
  } = props;

  const selected = projects.find((p) => p.id === selectedProjectId) ?? null;
  const projectJobs = jobs
    .filter((j) => j.experiment_id === selectedProjectId)
    .sort((a, b) => b.id - a.id);

  const [editingProjectId, setEditingProjectId] = useState<number | null>(null);
  const [editProjectName, setEditProjectName] = useState("");
  const [editingJobId, setEditingJobId] = useState<number | null>(null);
  const [editJobName, setEditJobName] = useState("");
  const [showCreateProject, setShowCreateProject] = useState(projects.length === 0);

  useEffect(() => {
    setEditingJobId(null);
    setEditJobName("");
  }, [selectedProjectId]);

  function selectProject(id: number) {
    onInvalidateBlueprint();
    onSelectedProjectIdChange(id);
  }

  return (
    <div className="project-jobs-hierarchy">
      <p className="hint hierarchy-intro">
        <strong>Project → Task → Job.</strong> Projects are long-lived. A{" "}
        <strong>task</strong> is a versioned JSON configuration. A <strong>job</strong> is one execution
        of that configuration.
      </p>

      <div className="project-jobs-columns project-jobs-columns-three">
        <section className="project-jobs-column card nested-card" aria-label="Projects">
          <header className="hierarchy-column-header">
            <h4>Projects</h4>
            <button
              type="button"
              className="btn-secondary btn-compact"
              onClick={() => setShowCreateProject((v) => !v)}
            >
              {showCreateProject ? "Cancel" : "+ New project"}
            </button>
          </header>

          {showCreateProject && (
            <div className="hierarchy-create-form">
              <label>
                Name
                <input
                  value={newProjectName}
                  onChange={(e) => onNewProjectNameChange(e.target.value)}
                  placeholder="e.g. Support chat models"
                />
              </label>
              <label>
                Description
                <input
                  value={newProjectDescription}
                  onChange={(e) => onNewProjectDescriptionChange(e.target.value)}
                  placeholder="Optional"
                />
              </label>
              <button
                type="button"
                disabled={!newProjectName.trim() || projectBusy}
                onClick={() => {
                  onCreateProject();
                  setShowCreateProject(false);
                }}
              >
                Create project
              </button>
            </div>
          )}

          {projects.length === 0 ? (
            <p className="hint">No projects yet. Create one to hold jobs and models.</p>
          ) : (
            <ul className="hierarchy-project-list">
              {projects.map((p) => {
                const active = p.id === selectedProjectId;
                const jobCount = jobs.filter((j) => j.experiment_id === p.id).length;
                return (
                  <li key={p.id}>
                    {editingProjectId === p.id ? (
                      <div className="hierarchy-inline-edit">
                        <input
                          value={editProjectName}
                          onChange={(e) => setEditProjectName(e.target.value)}
                        />
                        <button
                          type="button"
                          disabled={projectBusy || !editProjectName.trim()}
                          onClick={() => {
                            onUpdateProject(p.id, editProjectName.trim());
                            setEditingProjectId(null);
                          }}
                        >
                          Save
                        </button>
                        <button
                          type="button"
                          className="btn-secondary"
                          onClick={() => setEditingProjectId(null)}
                        >
                          Cancel
                        </button>
                      </div>
                    ) : (
                      <div className={`hierarchy-project-row ${active ? "active" : ""}`}>
                        <button
                          type="button"
                          className="hierarchy-project-select"
                          onClick={() => selectProject(p.id)}
                        >
                          <span className="hierarchy-project-name">{p.name}</span>
                          <span className="hint">
                            {jobCount} job{jobCount === 1 ? "" : "s"}
                          </span>
                        </button>
                        <span className="hierarchy-row-actions">
                          <button
                            type="button"
                            className="btn-secondary btn-compact"
                            onClick={() => {
                              setEditingProjectId(p.id);
                              setEditProjectName(p.name);
                            }}
                          >
                            Rename
                          </button>
                          <button
                            type="button"
                            className="btn-danger btn-compact"
                            disabled={projectBusy}
                            onClick={() => onDeleteProject(p.id)}
                          >
                            Delete
                          </button>
                        </span>
                      </div>
                    )}
                  </li>
                );
              })}
            </ul>
          )}
        </section>

        <section className="project-jobs-column card nested-card" aria-label="Task configurations">
          <header className="hierarchy-column-header">
            <h4>
              Tasks
              {selected ? (
                <span className="hint hierarchy-subtitle">in {selected.name}</span>
              ) : null}
            </h4>
          </header>
          {!selected ? (
            <p className="hint">Select a project to see task definitions.</p>
          ) : taskDefinitions.length === 0 ? (
            <p className="hint">
              No saved tasks yet. Use <strong>Save task version</strong> in the wizard or start a job
              (creates v1 automatically).
            </p>
          ) : (
            <ul className="hierarchy-project-list">
              {taskDefinitions.map((t) => {
                const active = t.id === selectedTaskId;
                return (
                  <li key={t.id}>
                    <button
                      type="button"
                      className={`hierarchy-project-select ${active ? "active" : ""}`}
                      onClick={() => onSelectedTaskIdChange?.(t.id)}
                    >
                      <span className="hierarchy-project-name">{t.name}</span>
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
          {selectedTaskId != null && taskVersions.length > 0 && (
            <div className="hierarchy-task-versions">
              <h5 className="hint">Versions</h5>
              <ul className="hierarchy-version-list">
                {taskVersions.map((v) => (
                  <li key={v.id}>
                    <button
                      type="button"
                      className="btn-secondary btn-compact"
                      title={`Hash ${v.content_hash.slice(0, 12)}…`}
                      onClick={() => onApplyTaskVersion?.(v.spec)}
                    >
                      v{v.version}
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </section>

        <section className="project-jobs-column card nested-card" aria-label="Job runs">
          <header className="hierarchy-column-header">
            <h4>
              Runs
              {selected ? (
                <span className="hint hierarchy-subtitle">in {selected.name}</span>
              ) : null}
            </h4>
          </header>

          {!selected ? (
            <p className="hint">Select a project to see job runs.</p>
          ) : projectJobs.length === 0 ? (
            <p className="hint">
              No runs yet. Complete the wizard and click{" "}
              <strong>Start job</strong> to queue training.
            </p>
          ) : (
            <ul className="hierarchy-job-list">
              {projectJobs.map((job) => (
                <li key={job.id} className="hierarchy-job-card">
                  {editingJobId === job.id ? (
                    <div className="hierarchy-inline-edit">
                      <input
                        value={editJobName}
                        onChange={(e) => setEditJobName(e.target.value)}
                      />
                      <button
                        type="button"
                        disabled={jobBusy || !editJobName.trim()}
                        onClick={() => {
                          onUpdateJobName(job.id, editJobName.trim());
                          setEditingJobId(null);
                        }}
                      >
                        Save
                      </button>
                      <button
                        type="button"
                        className="btn-secondary"
                        onClick={() => setEditingJobId(null)}
                      >
                        Cancel
                      </button>
                    </div>
                  ) : (
                    <>
                      <div className="hierarchy-job-head">
                        <strong>{job.name}</strong>
                        <span className={`pill job-status job-status-${job.status}`}>
                          {job.status}
                        </span>
                      </div>
                      <p className="hint hierarchy-job-meta">
                        Run #{job.id}
                        {job.task_version_id ? ` · task version id ${job.task_version_id}` : " · legacy"}
                        {job.openai_model_id ? ` · ${job.openai_model_id}` : ""}
                      </p>
                      <div className="hierarchy-row-actions">
                        <button
                          type="button"
                          className="btn-compact"
                          disabled={jobBusy || !canRunJob(job.status)}
                          onClick={() => onRunJob(job.id)}
                          title={
                            canRunJob(job.status)
                              ? "Queue this job to run again"
                              : "Stop the job or wait until it finishes"
                          }
                        >
                          Run
                        </button>
                        <button
                          type="button"
                          className="btn-secondary btn-compact"
                          disabled={jobBusy || !canStopJob(job.status)}
                          onClick={() => onStopJob(job.id)}
                          title={canStopJob(job.status) ? "Stop provisioning or training" : "Job is not running"}
                        >
                          Stop
                        </button>
                        <button
                          type="button"
                          className="btn-secondary btn-compact"
                          onClick={() => {
                            setEditingJobId(job.id);
                            setEditJobName(job.name);
                          }}
                        >
                          Rename
                        </button>
                        <button
                          type="button"
                          className="btn-danger btn-compact"
                          disabled={jobBusy}
                          onClick={() => onRequestDeleteJob(job.id)}
                        >
                          Delete
                        </button>
                      </div>
                    </>
                  )}
                </li>
              ))}
            </ul>
          )}
        </section>
      </div>
    </div>
  );
}
