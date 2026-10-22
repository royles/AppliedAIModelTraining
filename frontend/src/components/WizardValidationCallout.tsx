import type { WizardValidationMessage } from "../wizardValidationMessage";

type Props = {
  title?: string;
  block: WizardValidationMessage;
  id?: string;
  variant?: "step" | "footer";
};

export function WizardValidationCallout({
  title = "Complete this step to continue",
  block,
  id,
  variant = "step",
}: Props) {
  const className =
    variant === "footer" ? "wizard-footer-block-reason" : "wizard-step-validation";

  return (
    <div className={className} role="status" aria-live="polite" id={id}>
      {variant === "step" && <strong>{title}</strong>}
      {variant === "footer" && (
        <strong>{title === "Complete this step to continue" ? "Next is disabled" : title}</strong>
      )}
      <p className="wizard-validation-problem">{block.message}</p>
      <p className="wizard-validation-fix">
        <span className="wizard-validation-fix-label">Suggestion:</span> {block.fix}
      </p>
    </div>
  );
}
