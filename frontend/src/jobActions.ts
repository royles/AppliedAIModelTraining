/** Job run lifecycle helpers for Run / Stop / Delete UI. */

const ACTIVE_STATUSES = new Set(["queued", "provisioning", "running"]);

const RUNNABLE_STATUSES = new Set(["failed", "stopped", "succeeded"]);

export function isJobActive(status: string): boolean {
  return ACTIVE_STATUSES.has(status);
}

export function canRunJob(status: string): boolean {
  return RUNNABLE_STATUSES.has(status);
}

export function canStopJob(status: string): boolean {
  return isJobActive(status);
}

/** Prefer resume when re-running failed or user-stopped jobs. */
export function defaultResumeForRun(status: string): boolean {
  return status === "failed" || status === "stopped";
}
