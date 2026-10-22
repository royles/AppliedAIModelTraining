/**
 * GPT-style training pipeline. Architecture step blends editable geometry + live network diagram.
 */
export const MODEL_TASK_WIZARD_STEPS = [
  { id: "project", title: "Project" },
  { id: "mode", title: "Mode" },
  { id: "preset", title: "Model & task" },
  { id: "dataset", title: "Dataset" },
  { id: "tokenizer", title: "Tokenizer" },
  { id: "architecture", title: "Architecture" },
  { id: "training", title: "Schedule" },
  { id: "compute", title: "Compute" },
  { id: "register", title: "Register" },
] as const;

export type ModelTaskWizardStepId = (typeof MODEL_TASK_WIZARD_STEPS)[number]["id"];
