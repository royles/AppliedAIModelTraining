import type { ReactNode } from "react";

type Props = {
  stepId: string;
  title: string;
  children: ReactNode;
  canConfirm: boolean;
  confirmed: boolean;
  onConfirm: () => void;
  confirmHint?: string;
};

export function RecipeSection({
  stepId,
  title,
  children,
  canConfirm,
  confirmed,
  onConfirm,
  confirmHint,
}: Props) {
  return (
    <article
      id={`recipe-step-${stepId}`}
      className={`card recipe-section ${confirmed ? "recipe-section-confirmed" : ""}`}
    >
      <header className="recipe-section-header">
        <p className="recipe-step-label">Recipe step: {title}</p>
        <h3>{title}</h3>
      </header>
      {children}
      <div className="recipe-section-actions">
        <button
          type="button"
          className={confirmed ? "btn-secondary" : ""}
          disabled={!canConfirm}
          onClick={onConfirm}
        >
          {confirmed ? "Confirmed — edit to unlock" : "Confirm step"}
        </button>
        {!canConfirm && confirmHint && (
          <p className="hint recipe-section-hint">{confirmHint}</p>
        )}
      </div>
    </article>
  );
}
