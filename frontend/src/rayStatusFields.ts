export type RayStatusField = {
  id: string;
  label: string;
  value: string;
  tone?: "default" | "ok" | "warn" | "error";
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
  initialized: "Ray initialized (API process)",
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

/** Flatten ray_status JSON into stable, sorted name/value rows for the UI. */
export function rayStatusToFields(status: Record<string, unknown> | null | undefined): RayStatusField[] {
  if (!status) {
    return [];
  }

  const fields: RayStatusField[] = [];
  const seen = new Set<string>();

  for (const key of TOP_LEVEL_ORDER) {
    if (!(key in status)) {
      continue;
    }
    seen.add(key);
    const value = status[key];
    fields.push({
      id: key,
      label: humanizeKey(key),
      value: formatScalar(value),
      tone: toneForKey(key, value),
    });
  }

  const extraKeys = Object.keys(status)
    .filter((k) => k !== "cluster_resources" && !seen.has(k))
    .sort();
  for (const key of extraKeys) {
    fields.push({
      id: key,
      label: humanizeKey(key),
      value: formatScalar(status[key]),
      tone: toneForKey(key, status[key]),
    });
  }

  const resources = status.cluster_resources;
  if (resources && typeof resources === "object" && !Array.isArray(resources)) {
    const resourceKeys = Object.keys(resources as Record<string, unknown>).sort();
    for (const key of resourceKeys) {
      fields.push({
        id: `cluster_resources.${key}`,
        label: key,
        value: formatScalar((resources as Record<string, unknown>)[key]),
      });
    }
  }

  return fields;
}

/** Stable string for change detection (sorted keys). */
export function rayStatusFingerprint(status: Record<string, unknown> | null | undefined): string {
  return JSON.stringify(rayStatusToFields(status));
}
