/** Client-side wizard block — no network. */
export type WizardValidationMessage = {
  message: string;
  fix: string;
};

export function formatWizardValidation(msg: WizardValidationMessage): string {
  return `${msg.message} Suggestion: ${msg.fix}`;
}
