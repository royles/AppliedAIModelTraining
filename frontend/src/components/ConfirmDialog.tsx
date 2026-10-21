import { useEffect, useId, useRef, type ReactNode } from "react";

type Props = {
  open: boolean;
  title: string;
  children: ReactNode;
  confirmLabel?: string;
  cancelLabel?: string;
  danger?: boolean;
  busy?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
};

export function ConfirmDialog({
  open,
  title,
  children,
  confirmLabel = "Confirm",
  cancelLabel = "Cancel",
  danger = false,
  busy = false,
  onConfirm,
  onCancel,
}: Props) {
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
      <form
        method="dialog"
        className="cdl-dialog-panel"
        onSubmit={(e) => {
          e.preventDefault();
          if (!busy) {
            onConfirm();
          }
        }}
      >
        <header className="cdl-dialog-header">
          <h2 id={titleId} className="cdl-dialog-title">
            {title}
          </h2>
        </header>
        <div id={descId} className="cdl-dialog-body">
          {children}
        </div>
        <footer className="cdl-dialog-footer">
          <button ref={cancelRef} type="button" className="btn-secondary" disabled={busy} onClick={onCancel}>
            {cancelLabel}
          </button>
          <button type="submit" className={danger ? "btn-danger" : undefined} disabled={busy}>
            {busy ? "Working…" : confirmLabel}
          </button>
        </footer>
      </form>
    </dialog>
  );
}
