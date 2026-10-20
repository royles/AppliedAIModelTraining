import { useEffect, useState } from "react";
import type { Project, TrainingJob } from "./api";

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
  onDeleteJob: (jobId: number) => void;
  onRerunJob: (jobId: number) => void;
  projectBusy: boolean;
  jobBusy: boolean;
  onInvalidateBlueprint: () => void;
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
    onDeleteJob,
    onRerunJob,
    projectBusy,
    jobBusy,
    onInvalidateBlueprint,
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
        <strong>Project → Jobs.</strong> A project is a collection of models. Each job defines a
        training run; re-run a job to train a new model version (same blueprint settings unless you
        change them below).
      </p>

      <div className="project-jobs-columns">
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

        <section className="project-jobs-column card nested-card" aria-label="Jobs">
          <header className="hierarchy-column-header">
            <h4>
              Jobs
              {selected ? (
                <span className="hint hierarchy-subtitle">in {selected.name}</span>
              ) : null}
            </h4>
          </header>

          {!selected ? (
            <p className="hint">Select a project to see its jobs.</p>
          ) : projectJobs.length === 0 ? (
            <p className="hint">
              No jobs in this project yet. Complete the blueprint below and click{" "}
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
                        Job #{job.id}
                        {job.openai_model_id ? ` · model id ${job.openai_model_id}` : ""}
                      </p>
                      <div className="hierarchy-row-actions">
                        <button
                          type="button"
                          disabled={jobBusy}
                          onClick={() => onRerunJob(job.id)}
                          title="Queue another training run for this job"
                        >
                          Re-run job
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
                          onClick={() => onDeleteJob(job.id)}
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
