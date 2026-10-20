import { useEffect, useState } from "react";
import type { Project, TrainingJob } from "./api";

type Props = {
  projects: Project[];
  selectedProjectId: number | null;
  onSelectedProjectIdChange: (id: number) => void;
  newProjectName: string;
  onNewProjectNameChange: (v: string) => void;
  newProjectDescription: string;
  onNewProjectDescriptionChange: (v: string) => void;
  onCreateProject: () => void;
  onUpdateProject: (projectId: number, name: string) => void;
  onDeleteProject: (projectId: number) => void;
  projectBusy: boolean;
  jobs: TrainingJob[];
  onDeleteJob: (jobId: number) => void;
  onUpdateJobName: (jobId: number, name: string) => void;
  jobBusy: boolean;
  onInvalidateBlueprint: () => void;
};

export function ProjectTasksFields(props: Props) {
  const {
    projects,
    selectedProjectId,
    onSelectedProjectIdChange,
    newProjectName,
    onNewProjectNameChange,
    newProjectDescription,
    onNewProjectDescriptionChange,
    onCreateProject,
    onUpdateProject,
    onDeleteProject,
    projectBusy,
    jobs,
    onDeleteJob,
    onUpdateJobName,
    jobBusy,
    onInvalidateBlueprint,
  } = props;

  const selected = projects.find((p) => p.id === selectedProjectId) ?? null;
  const projectJobs = jobs.filter((j) => j.experiment_id === selectedProjectId);

  const [renameProject, setRenameProject] = useState("");
  const [editingJobId, setEditingJobId] = useState<number | null>(null);
  const [editJobName, setEditJobName] = useState("");

  useEffect(() => {
    setRenameProject(selected?.name ?? "");
    setEditingJobId(null);
    setEditJobName("");
  }, [selected?.id, selected?.name]);

  return (
    <div className="project-tasks-fields">
      <div className="project-tasks-create">
        <label>
          New project name
          <input
            value={newProjectName}
            onChange={(e) => onNewProjectNameChange(e.target.value)}
            placeholder="e.g. Customer support models"
          />
        </label>
        <label>
          Description (optional)
          <input
            value={newProjectDescription}
            onChange={(e) => onNewProjectDescriptionChange(e.target.value)}
            placeholder="Short note for your team"
          />
        </label>
        <button
          type="button"
          disabled={!newProjectName.trim() || projectBusy}
          onClick={onCreateProject}
        >
          Create project
        </button>
      </div>

      <label>
        Active project
        <select
          value={selectedProjectId ?? ""}
          onChange={(e) => {
            onInvalidateBlueprint();
            onSelectedProjectIdChange(Number(e.target.value));
          }}
          disabled={projects.length === 0}
        >
          {projects.length === 0 ? (
            <option value="">No projects yet</option>
          ) : (
            projects.map((p) => (
              <option key={p.id} value={p.id}>{p.name}</option>
            ))
          )}
        </select>
      </label>

      {selected && (
        <div className="project-tasks-manage card nested-card">
          <h4>Manage “{selected.name}”</h4>
          <label>
            Rename project
            <input
              value={renameProject}
              onChange={(e) => setRenameProject(e.target.value)}
            />
          </label>
          <div className="button-row">
            <button
              type="button"
              disabled={projectBusy || !renameProject.trim()}
              onClick={() => onUpdateProject(selected.id, renameProject.trim())}
            >
              Save project
            </button>
            <button
              type="button"
              className="btn-danger"
              disabled={projectBusy}
              onClick={() => onDeleteProject(selected.id)}
            >
              Delete project
            </button>
          </div>
        </div>
      )}

      <div className="project-tasks-jobs">
        <h4>Tasks in this project</h4>
        <p className="hint">
          Tasks are training jobs stored in the database. Configure the blueprint below and use{" "}
          <strong>Start model task</strong> to queue a new one.
        </p>
        {selectedProjectId === null ? (
          <p className="hint">Select or create a project first.</p>
        ) : projectJobs.length === 0 ? (
          <p className="hint">No tasks yet in this project.</p>
        ) : (
          <ul className="project-job-list">
            {projectJobs.map((job) => (
              <li key={job.id}>
                {editingJobId === job.id ? (
                  <div className="project-job-edit">
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
                  <div className="project-job-row">
                    <span>
                      <strong>{job.name}</strong>
                      <span className="hint"> · {job.status}</span>
                      {job.openai_model_id && (
                        <span className="hint"> · id {job.openai_model_id}</span>
                      )}
                    </span>
                    <span className="project-job-actions">
                      <button
                        type="button"
                        onClick={() => {
                          setEditingJobId(job.id);
                          setEditJobName(job.name);
                        }}
                      >
                        Rename
                      </button>
                      <button
                        type="button"
                        className="btn-danger"
                        disabled={jobBusy}
                        onClick={() => onDeleteJob(job.id)}
                      >
                        Delete
                      </button>
                    </span>
                  </div>
                )}
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
