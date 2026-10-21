import { useCallback, useId, useRef, useState, type ReactNode } from "react";

type Props = {
  label: string;
  children: ReactNode;
  className?: string;
};

/** Accessible tooltip: hover + focus, Escape dismisses. */
export function Tooltip({ label, children, className }: Props) {
  const id = useId();
  const [open, setOpen] = useState(false);
  const wrapRef = useRef<HTMLSpanElement>(null);

  const show = useCallback(() => setOpen(true), []);
  const hide = useCallback(() => setOpen(false), []);

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Escape") {
      setOpen(false);
      wrapRef.current?.querySelector<HTMLElement>("button, a, [tabindex]")?.focus();
    }
  };

  return (
    <span
      ref={wrapRef}
      className={`cdl-tooltip-wrap ${className ?? ""}`.trim()}
      onMouseEnter={show}
      onMouseLeave={hide}
      onFocus={show}
      onBlur={hide}
      onKeyDown={onKeyDown}
    >
      <span aria-describedby={open ? id : undefined}>{children}</span>
      {open && (
        <span id={id} role="tooltip" className="cdl-tooltip">
          {label}
        </span>
      )}
    </span>
  );
}
