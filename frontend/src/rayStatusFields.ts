export type RayStatusField = {
  id: string;
  label: string;
  value: string;
  tone?: "default" | "ok" | "warn" | "error";
};

export type RayStatusStat = {
  id: string;
  label: string;
  value: string;
  numeric?: number;
  tone?: "default" | "ok" | "warn" | "error";
  format?: "pill" | "code" | "text";
};

export type RayResourceMetric = {
  id: string;
  label: string;
  value: number;
  display: string;
  accent: "cpu" | "gpu" | "memory" | "node" | "default";
};

export type RayStatusDashboard = {
  stats: RayStatusStat[];
  resources: RayResourceMetric[];
  messages: RayStatusField[];
};

const TOP_LEVEL_ORDER = [
  "job_ray_address",
  "initialized",
  "runtime",
  "nodes",
  "cluster_error",
  "note",
] as const;

const LABELS: Record<string, string> = {
  job_ray_address: "Job Ray address",
  initialized: "Ray initialized",
  runtime: "Runtime",
  nodes: "Nodes",
  cluster_error: "Cluster error",
  note: "Note",
};

function humanizeKey(key: string): string {
  return LABELS[key] ?? key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function formatScalar(value: unknown): string {
  if (value === null || value === undefined) {
    return "—";
  }
  if (typeof value === "boolean") {
    return value ? "Yes" : "No";
  }
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toLocaleString();
  }
  if (typeof value === "string") {
    return value;
  }
  return JSON.stringify(value);
}

function parseNumeric(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string" && value.trim() !== "") {
    const n = Number(value);
    return Number.isFinite(n) ? n : null;
  }
  return null;
}

function toneForKey(key: string, value: unknown): RayStatusField["tone"] {
  if (key === "initialized") {
    return value === true ? "ok" : "warn";
  }
  if (key === "cluster_error") {
    return "error";
  }
  if (key === "note") {
    return "warn";
  }
  return "default";
}

function resourceAccent(key: string): RayResourceMetric["accent"] {
  const k = key.toLowerCase();
  if (k === "cpu" || k.includes("cpu")) {
    return "cpu";
  }
  if (k === "gpu" || k.includes("gpu")) {
    return "gpu";
  }
  if (k.includes("memory") || k === "mem") {
    return "memory";
  }
  if (k === "node" || k.startsWith("node:")) {
    return "node";
  }
  return "default";
}

/** Flatten ray_status JSON into stable, sorted name/value rows for the UI. */
export function rayStatusToFields(status: Record<string, unknown> | null | undefined): RayStatusField[] {
  const dash = rayStatusToDashboard(status);
  const rows: RayStatusField[] = [];
  for (const s of dash.stats) {
    rows.push({ id: s.id, label: s.label, value: s.value, tone: s.tone });
  }
  for (const r of dash.resources) {
    rows.push({ id: r.id, label: r.label, value: r.display });
  }
  return rows.concat(dash.messages);
}

export function rayStatusToDashboard(
  status: Record<string, unknown> | null | undefined,
): RayStatusDashboard {
  if (!status) {
    return { stats: [], resources: [], messages: [] };
  }

  const stats: RayStatusStat[] = [];
  const messages: RayStatusField[] = [];
  const seen = new Set<string>();

  for (const key of TOP_LEVEL_ORDER) {
    if (!(key in status)) {
      continue;
    }
    seen.add(key);
    const raw = status[key];
    if (key === "cluster_error" || key === "note") {
      messages.push({
        id: key,
        label: humanizeKey(key),
        value: formatScalar(raw),
        tone: toneForKey(key, raw),
      });
      continue;
    }
    if (key === "job_ray_address") {
      stats.push({
        id: key,
        label: humanizeKey(key),
        value: formatScalar(raw),
        format: "code",
      });
      continue;
    }
    if (key === "initialized") {
      stats.push({
        id: key,
        label: humanizeKey(key),
        value: formatScalar(raw),
        tone: toneForKey(key, raw),
        format: "pill",
      });
      continue;
    }
    const numeric = parseNumeric(raw);
    stats.push({
      id: key,
      label: humanizeKey(key),
      value: formatScalar(raw),
      numeric: numeric ?? undefined,
      format: "text",
    });
  }

  const extraKeys = Object.keys(status)
    .filter((k) => k !== "cluster_resources" && !seen.has(k))
    .sort();
  for (const key of extraKeys) {
    const raw = status[key];
    const numeric = parseNumeric(raw);
    if (numeric !== null && key !== "job_ray_address") {
      stats.push({
        id: key,
        label: humanizeKey(key),
        value: formatScalar(raw),
        numeric,
      });
    } else {
      stats.push({
        id: key,
        label: humanizeKey(key),
        value: formatScalar(raw),
        format: key.includes("address") ? "code" : "text",
      });
    }
  }

  const resources: RayResourceMetric[] = [];
  const resourceObj = status.cluster_resources;
  if (resourceObj && typeof resourceObj === "object" && !Array.isArray(resourceObj)) {
    const resourceKeys = Object.keys(resourceObj as Record<string, unknown>).sort();
    for (const key of resourceKeys) {
      const raw = (resourceObj as Record<string, unknown>)[key];
      const numeric = parseNumeric(raw);
      if (numeric === null) {
        continue;
      }
      resources.push({
        id: `cluster_resources.${key}`,
        label: key,
        value: numeric,
        display: formatScalar(raw),
        accent: resourceAccent(key),
      });
    }
  }

  return { stats, resources, messages };
}

/** Max scale for bar charts (largest metric in the snapshot). */
export function rayResourceBarMax(
  dashboard: RayStatusDashboard,
  jobWorkers?: number,
): number {
  const values: number[] = dashboard.resources.map((r) => r.value);
  for (const s of dashboard.stats) {
    if (s.numeric !== undefined) {
      values.push(s.numeric);
    }
  }
  if (jobWorkers !== undefined && jobWorkers > 0) {
    values.push(jobWorkers);
  }
  return Math.max(1, ...values, 0);
}

/** Stable string for change detection (sorted keys). */
export function rayStatusFingerprint(status: Record<string, unknown> | null | undefined): string {
  return JSON.stringify(rayStatusToDashboard(status));
}
