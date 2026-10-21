import type { ButtonHTMLAttributes, ReactNode } from "react";
import { Tooltip } from "./Tooltip";

type Props = ButtonHTMLAttributes<HTMLButtonElement> & {
  /** Required for icon-only buttons (WCAG). */
  ariaLabel: string;
  tooltip?: string;
  children: ReactNode;
};

export function IconButton({
  ariaLabel,
  tooltip,
  children,
  className,
  type = "button",
  ...rest
}: Props) {
  const btn = (
    <button
      type={type}
      className={`cdl-icon-btn ${className ?? ""}`.trim()}
      aria-label={ariaLabel}
      {...rest}
    >
      {children}
    </button>
  );
  if (tooltip && tooltip !== ariaLabel) {
    return <Tooltip label={tooltip}>{btn}</Tooltip>;
  }
  return btn;
}
