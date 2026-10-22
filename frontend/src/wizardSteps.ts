/**
 * GPT-style training pipeline: project → mode → model/task → data → tokenizer →
 * architecture → schedule → cluster → register. One screen per step (split, don’t scroll).
 */
export const MODEL_TASK_WIZARD_STEPS = [
  { id: "project", title: "Project" },
  { id: "mode", title: "Mode" },
  { id: "preset", title: "Model & task" },
  { id: "dataset", title: "Dataset" },
  { id: "tokenizer", title: "Tokenizer" },
  { id: "geometry", title: "Geometry" },
  { id: "diagram", title: "Diagram" },
  { id: "training", title: "Schedule" },
  { id: "ray", title: "Ray defaults" },
  { id: "workers", title: "Workers" },
  { id: "register", title: "Register" },
] as const;

export type ModelTaskWizardStepId = (typeof MODEL_TASK_WIZARD_STEPS)[number]["id"];
