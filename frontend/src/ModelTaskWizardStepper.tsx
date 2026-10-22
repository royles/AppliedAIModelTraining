import type { BlueprintStep } from "./trainingBlueprint";

type WizardStepDef = { id: string; title: string };

type Props = {
  steps: WizardStepDef[];
  blueprintSteps: BlueprintStep[];
  currentIndex: number;
  onSelectIndex: (index: number) => void;
};

export function ModelTaskWizardStepper({
  steps,
  blueprintSteps,
  currentIndex,
  onSelectIndex,
}: Props) {
  const doneById = new Map(blueprintSteps.map((s) => [s.id, s.done]));

  return (
    <nav className="model-task-wizard-stepper" aria-label="Model task wizard steps">
      <ol className="model-task-wizard-stepper-list">
        {steps.map((step, index) => {
          const done = doneById.get(step.id) ?? false;
          const active = index === currentIndex;
          return (
            <li key={step.id} className="model-task-wizard-stepper-item">
              <button
                type="button"
                className={`model-task-wizard-step ${active ? "active" : ""} ${done ? "done" : ""}`}
                aria-current={active ? "step" : undefined}
                onClick={() => onSelectIndex(index)}
              >
                <span className="model-task-wizard-step-num">{index + 1}</span>
                <span className="model-task-wizard-step-title">{step.title}</span>
              </button>
              {index < steps.length - 1 && (
                <span className="model-task-wizard-step-connector" aria-hidden />
              )}
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
