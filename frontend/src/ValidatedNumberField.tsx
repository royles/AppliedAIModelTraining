import { useEffect, useState } from "react";

type Props = {
  id?: string;
  "aria-describedby"?: string;
  value: number;
  onChange: (n: number) => void;
  min?: number;
  max?: number;
  integer?: boolean;
  disabled?: boolean;
};

export function ValidatedNumberField({
  id,
  "aria-describedby": ariaDescribedBy,
  value,
  onChange,
  min,
  max,
  integer = true,
  disabled,
}: Props) {
  const [text, setText] = useState(String(value));
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setText(String(value));
    setError(null);
  }, [value]);

  function commit() {
    const trimmed = text.trim();
    if (!trimmed) {
      setError("Enter a number");
      setText(String(value));
      return;
    }
    const n = integer ? Number.parseInt(trimmed, 10) : Number.parseFloat(trimmed);
    if (!Number.isFinite(n) || (integer && !/^-?\d+$/.test(trimmed))) {
      setError("Enter a valid number");
      setText(String(value));
      return;
    }
    if (min !== undefined && n < min) {
      setError(`Minimum is ${min}`);
      setText(String(value));
      return;
    }
    if (max !== undefined && n > max) {
      setError(`Maximum is ${max}`);
      setText(String(value));
      return;
    }
    setError(null);
    if (n !== value) {
      onChange(n);
    }
  }

  return (
    <span className="validated-number-field">
      <input
        id={id}
        type="text"
        inputMode={integer ? "numeric" : "decimal"}
        disabled={disabled}
        value={text}
        aria-invalid={error ? true : undefined}
        aria-describedby={ariaDescribedBy}
        className={error ? "input-invalid" : undefined}
        onChange={(e) => {
          setText(e.target.value);
          setError(null);
        }}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            e.preventDefault();
            commit();
          }
        }}
      />
      {error && <span className="field-error">{error}</span>}
    </span>
  );
}
