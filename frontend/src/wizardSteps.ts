export const MODEL_TASK_WIZARD_STEPS = [
  { id: "project", title: "Project" },
  { id: "mode", title: "Mode" },
  { id: "dataset", title: "Dataset" },
  { id: "tokenizer", title: "Tokenizer" },
  { id: "architecture", title: "Architecture" },
  { id: "training", title: "Training" },
  { id: "compute", title: "Compute" },
  { id: "register", title: "Register" },
] as const;

export type ModelTaskWizardStepId = (typeof MODEL_TASK_WIZARD_STEPS)[number]["id"];
