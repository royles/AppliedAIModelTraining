export function formatLogEntry(ts: string, message: string): string {
  const prefix = `[${ts}] `;
  const parts = message.split("\n");
  if (parts.length <= 1) {
    return prefix + message;
  }
  const indent = " ".repeat(prefix.length);
  return parts.map((line, i) => (i === 0 ? prefix + line : indent + line)).join("\n");
}

export function formatLogLines(lines: { ts: string; message: string }[]): string {
  if (lines.length === 0) return "No entries yet.";
  return lines.map((l) => formatLogEntry(l.ts, l.message)).join("\n");
}

/** Box-drawing chars in framework banners should share the same column on every row. */
export function boxColumnAligned(formatted: string, boxStartColumn: number): boolean {
  const rows = formatted.split("\n");
  const boxChars = new Set("╔╗╚╝║═");
  const columns: number[] = [];
  for (const row of rows) {
    for (let i = 0; i < row.length; i++) {
      if (boxChars.has(row[i])) {
        columns.push(i);
        break;
      }
    }
  }
  if (columns.length === 0) {
    return true;
  }
  return columns.every((c) => c === boxStartColumn);
}
