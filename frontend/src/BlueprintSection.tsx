import type { ReactNode } from "react";

type Props = {
  stepId: string;
  title: string;
  description?: string;
  children: ReactNode;
  canConfirm: boolean;
  confirmed: boolean;
  onConfirm: () => void;
  confirmHint?: string;
};

export function BlueprintSection({
  stepId,
  title,
  description,
  children,
  canConfirm,
  confirmed,
  onConfirm,
  confirmHint,
}: Props) {
  return (
    <article
      id={`blueprint-step-${stepId}`}
      className={`card blueprint-section ${confirmed ? "blueprint-section-confirmed" : ""}`}
    >
      <header className="blueprint-section-header">
        <p className="blueprint-step-label">Blueprint step</p>
        <h3>{title}</h3>
        {description && <p className="blueprint-step-description">{description}</p>}
      </header>
      {children}
      <div className="blueprint-section-actions">
        <button
          type="button"
          className={confirmed ? "btn-secondary" : ""}
          disabled={!canConfirm}
          onClick={onConfirm}
        >
          {confirmed ? "Confirmed — edit to unlock" : "Confirm step"}
        </button>
        {!canConfirm && confirmHint && (
          <p className="hint blueprint-section-hint">{confirmHint}</p>
        )}
      </div>
    </article>
  );
}
