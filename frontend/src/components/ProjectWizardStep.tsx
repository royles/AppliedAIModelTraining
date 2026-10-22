import type { Project } from "../api";

type Props = {
  projects: Project[];
  selectedProjectId: number | null;
  onSelectedProjectIdChange: (id: number) => void;
  newProjectName: string;
  onNewProjectNameChange: (v: string) => void;
  newProjectDescription: string;
  onNewProjectDescriptionChange: (v: string) => void;
  onCreateProject: () => void;
  projectBusy: boolean;
};

export function ProjectWizardStep({
  projects,
  selectedProjectId,
  onSelectedProjectIdChange,
  newProjectName,
  onNewProjectNameChange,
  newProjectDescription,
  onNewProjectDescriptionChange,
  onCreateProject,
  projectBusy,
}: Props) {
  return (
    <div className="wizard-step-compact">
      <label className="architecture-field">
        <span className="architecture-field-label">Active project</span>
        <select
          value={selectedProjectId ?? ""}
          onChange={(e) => {
            const id = Number(e.target.value);
            if (Number.isFinite(id)) {
              onSelectedProjectIdChange(id);
            }
          }}
        >
          <option value="" disabled>
            Select a project…
          </option>
          {projects.map((p) => (
            <option key={p.id} value={p.id}>
              {p.name}
            </option>
          ))}
        </select>
      </label>
      <fieldset className="wizard-inline-fieldset">
        <legend>New project</legend>
        <div className="architecture-form-grid architecture-form-grid-2">
          <label className="architecture-field">
            <span className="architecture-field-label">Name</span>
            <input
              value={newProjectName}
              onChange={(e) => onNewProjectNameChange(e.target.value)}
              placeholder="my-training-project"
            />
          </label>
          <label className="architecture-field">
            <span className="architecture-field-label">Description (optional)</span>
            <input
              value={newProjectDescription}
              onChange={(e) => onNewProjectDescriptionChange(e.target.value)}
            />
          </label>
        </div>
        <button type="button" className="btn-secondary" disabled={projectBusy} onClick={onCreateProject}>
          {projectBusy ? "Creating…" : "Create project"}
        </button>
      </fieldset>
      <p className="hint wizard-step-footnote">
        Manage existing jobs and task versions from the <strong>Tasks</strong> view in the sidebar.
      </p>
    </div>
  );
}
