import { useEffect, useId, useRef } from "react";

type Props = {
  open: boolean;
  jobName: string;
  busy?: boolean;
  onResume: () => void;
  onStartOver: () => void;
  onCancel: () => void;
};

export function JobRestartDialog({ open, jobName, busy, onResume, onStartOver, onCancel }: Props) {
  const titleId = useId();
  const descId = useId();
  const dialogRef = useRef<HTMLDialogElement>(null);
  const cancelRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    const el = dialogRef.current;
    if (!el) return;
    if (open && !el.open) {
      el.showModal();
      cancelRef.current?.focus();
    } else if (!open && el.open) {
      el.close();
    }
  }, [open]);

  useEffect(() => {
    const el = dialogRef.current;
    if (!el) return;
    const onCancelEvent = (e: Event) => {
      e.preventDefault();
      onCancel();
    };
    el.addEventListener("cancel", onCancelEvent);
    return () => el.removeEventListener("cancel", onCancelEvent);
  }, [onCancel]);

  return (
    <dialog
      ref={dialogRef}
      className="cdl-dialog"
      aria-labelledby={titleId}
      aria-describedby={descId}
      onClick={(e) => {
        if (e.target === dialogRef.current) {
          onCancel();
        }
      }}
    >
      <div className="cdl-dialog-panel">
        <header className="cdl-dialog-header">
          <h2 id={titleId} className="cdl-dialog-title">
            Run job again
          </h2>
        </header>
        <div id={descId} className="cdl-dialog-body">
          <p>
            Run <strong>{jobName}</strong> again with the same parameters?
          </p>
          <p className="hint">
            Resume continues from the last saved epoch/step when available. Start over runs a fresh training
            loop but keeps prior artifacts on disk.
          </p>
        </div>
        <footer className="cdl-dialog-footer">
          <button ref={cancelRef} type="button" className="btn-secondary" disabled={busy} onClick={onCancel}>
            Cancel
          </button>
          <button type="button" className="btn-secondary" disabled={busy} onClick={onStartOver}>
            Start over
          </button>
          <button type="button" disabled={busy} onClick={onResume}>
            {busy ? "Starting…" : "Resume"}
          </button>
        </footer>
      </div>
    </dialog>
  );
}
