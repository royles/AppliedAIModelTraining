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

/** Framework provisioning then training — one chronological stream for the task detail log. */
const RAY_TRAIN_NOISE = [
  /Failed to query Ray Train Controller actor state/i,
  /`ray\.train\.report` has not been called by all \d+ workers/i,
  /RAY_TRAIN_COLLECTIVE_WARN_INTERVAL_S/,
  /following ranks to join the `ray\.train\.report` call/i,
];

function isNoiseLogLine(line: string): boolean {
  const text = line.trim();
  if (!text) return true;
  if (/^\(PlacementGroupCleaner pid=\d+\)\s*$/i.test(text)) return true;
  if (/^\(SynchronizationActor pid=\d+\)\s*$/i.test(text)) return true;
  return RAY_TRAIN_NOISE.some((re) => re.test(text));
}

export function isRayTrainNoiseMessage(message: string): boolean {
  const lines = message.split("\n").map((l) => l.trim()).filter(Boolean);
  if (lines.length === 0) return true;
  return lines.every((line) => isNoiseLogLine(line));
}

export function mergeSerialJobLogs(
  framework: { ts: string; message: string }[],
  training: { ts: string; message: string }[],
): { ts: string; message: string }[] {
  const merged = [...framework, ...training].filter((line) => !isRayTrainNoiseMessage(line.message));
  merged.sort((a, b) => {
    const t = a.ts.localeCompare(b.ts);
    if (t !== 0) return t;
    return a.message.localeCompare(b.message);
  });
  return merged;
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
