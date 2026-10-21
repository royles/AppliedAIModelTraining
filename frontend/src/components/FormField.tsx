import type { ReactNode } from "react";
import { Tooltip } from "./Tooltip";

type Props = {
  id: string;
  label: ReactNode;
  hint?: string;
  error?: string | null;
  tooltip?: string;
  required?: boolean;
  children: ReactNode;
};

export function FormField({ id, label, hint, error, tooltip, required, children }: Props) {
  const hintId = hint ? `${id}-hint` : undefined;
  const errorId = error ? `${id}-error` : undefined;
  const describedBy = [hintId, errorId].filter(Boolean).join(" ") || undefined;

  return (
    <div className={`cdl-field ${error ? "cdl-field-error" : ""}`}>
      <div className="cdl-field-label-row">
        <label htmlFor={id} className="cdl-field-label">
          {label}
          {required && (
            <span className="cdl-required" aria-hidden>
              {" "}
              *
            </span>
          )}
          {required && <span className="visually-hidden"> (required)</span>}
        </label>
        {tooltip && (
          <Tooltip label={tooltip}>
            <button type="button" className="cdl-field-help-btn" aria-label={`Help: ${tooltip}`}>
              ?
            </button>
          </Tooltip>
        )}
      </div>
      <div aria-describedby={describedBy}>{children}</div>
      {hint && !error && (
        <p id={hintId} className="hint cdl-field-hint">
          {hint}
        </p>
      )}
      {error && (
        <p id={errorId} className="cdl-field-error-msg" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}
