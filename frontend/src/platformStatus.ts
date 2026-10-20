import type { Health, TrainingJob } from "./api";

export type IndicatorTone = "loading" | "ok" | "warn" | "error" | "idle";

export type PlatformRayStatus = {
  gcs_address?: string | null;
  cdsw_available?: boolean;
  gcs_reachable?: boolean;
  alive_nodes?: number | null;
  gpu_capacity?: number | null;
  state?: string;
};

export type ActiveJobsSummary = {
  count: number;
  jobs: { id: number; name: string; status: string }[];
};

export const ACTIVE_JOB_STATUSES = new Set(["queued", "provisioning", "running"]);

export function activeJobsFromList(jobs: TrainingJob[]): TrainingJob[] {
  return jobs.filter((j) => ACTIVE_JOB_STATUSES.has(j.status));
}

export function parsePlatformRay(details: Record<string, unknown> | undefined): PlatformRayStatus | null {
  const raw = details?.platform_ray;
  if (!raw || typeof raw !== "object") return null;
  return raw as PlatformRayStatus;
}

export function parseActiveJobsSummary(
  details: Record<string, unknown> | undefined,
): ActiveJobsSummary | null {
  const raw = details?.active_jobs;
  if (!raw || typeof raw !== "object") return null;
  const o = raw as ActiveJobsSummary;
  if (typeof o.count !== "number" || !Array.isArray(o.jobs)) return null;
  return o;
}

export function rayIndicatorFromHealth(health: Health | null, apiOk: boolean): {
  tone: IndicatorTone;
  label: string;
  title: string;
} {
  if (!apiOk) {
    return { tone: "loading", label: "Ray", title: "Waiting for API…" };
  }
  const ray = parsePlatformRay(health?.details);
  if (!ray) {
    return { tone: "warn", label: "Ray", title: "Cluster status unavailable" };
  }
  const addr = ray.gcs_address ?? "—";
  if (ray.state === "not_configured") {
    return {
      tone: ray.cdsw_available ? "warn" : "idle",
      label: "Ray",
      title: ray.cdsw_available
        ? "Ray head not configured yet"
        : "Local mode — no CDSW Ray cluster",
    };
  }
  if (ray.state === "local") {
    return {
      tone: "idle",
      label: "Ray local",
      title: "Training uses an in-process Ray runtime on this machine",
    };
  }
  if (ray.state === "provisioning") {
    return {
      tone: "warn",
      label: "Ray (task)",
      title: `A model task is starting or running Ray on ${addr ?? "session"} — see task framework log`,
    };
  }
  if (!ray.gcs_reachable) {
    if (ray.state === "deferred") {
      return {
        tone: "idle",
        label: "Ray idle",
        title: `Cluster starts when you run a model task (head will be ${addr})`,
      };
    }
    return {
      tone: "warn",
      label: "Ray starting",
      title: `GCS not reachable at ${addr}. Workers may still be joining.`,
    };
  }
  const nodes = ray.alive_nodes;
  const gpu = ray.gpu_capacity;
  if (nodes != null && nodes >= 1) {
    const gpuBit = gpu != null && gpu > 0 ? ` · ${gpu} GPU` : "";
    return {
      tone: "ok",
      label: `Ray online (${nodes} node${nodes === 1 ? "" : "s"}${gpuBit})`,
      title: `Cluster reachable at ${addr}`,
    };
  }
  return {
    tone: "warn",
    label: "Ray reachable",
    title: `GCS up at ${addr}; waiting for worker nodes`,
  };
}

export function tasksIndicator(
  jobs: TrainingJob[],
  summary: ActiveJobsSummary | null,
): { tone: IndicatorTone; label: string; title: string } {
  const count = summary?.count ?? activeJobsFromList(jobs).length;
  if (count <= 0) {
    return { tone: "idle", label: "No active tasks", title: "No jobs queued or running" };
  }
  const list = summary?.jobs?.length
    ? summary.jobs
    : activeJobsFromList(jobs).map((j) => ({ id: j.id, name: j.name, status: j.status }));
  const running = list.filter((j) => j.status === "running").length;
  const provisioning = list.filter((j) => j.status === "provisioning").length;
  const queued = list.filter((j) => j.status === "queued").length;
  const parts: string[] = [];
  if (running) parts.push(`${running} running`);
  if (provisioning) parts.push(`${provisioning} provisioning`);
  if (queued) parts.push(`${queued} queued`);
  const label = parts.length ? parts.join(", ") : `${count} active`;
  const title = list.map((j) => `#${j.id} ${j.name} (${j.status})`).join("\n");
  return { tone: running > 0 || provisioning > 0 ? "ok" : "warn", label, title };
}
