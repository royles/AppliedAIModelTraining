import { useEffect, useState } from "react";
import {
  api,
  type CdswWorkerSettings,
  type PlatformSettings,
  type RaySettings,
} from "./api";
import { WorkerSizePresetBar } from "./components/WorkerSizePresetBar";
import type { WorkerSizePreset } from "./workerSizePresets";

type Props = {
  /** When set, load/save Ray config for this project (1:1 with the project). */
  projectId?: number;
  onSaved?: () => void;
  title?: string;
  saveLabel?: string;
  introHint?: string;
};

export function PlatformSettingsPanel({
  projectId,
  onSaved,
  title = "Framework Configuration",
  saveLabel,
  introHint,
}: Props) {
  const [settings, setSettings] = useState<PlatformSettings | null>(null);
  const [ray, setRay] = useState<RaySettings | null>(null);
  const [cdsw, setCdsw] = useState<CdswWorkerSettings | null>(null);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [workerPresetId, setWorkerPresetId] = useState<string | null>(null);
  const isProjectScope = projectId != null;

  function applyWorkerPreset(p: WorkerSizePreset) {
    if (!cdsw) return;
    setWorkerPresetId(p.id);
    setCdsw({
      ...cdsw,
      worker_cpu: p.worker_cpu,
      worker_memory_gb: p.worker_memory_gb,
      gpus_per_worker: p.gpus_per_worker,
    });
  }

  useEffect(() => {
    let cancelled = false;
    setSettings(null);
    setRay(null);
    setCdsw(null);
    setMessage(null);
    const load = isProjectScope
      ? api.getProjectRaySettings(projectId)
      : api.getPlatformSettings();
    load.then((s) => {
      if (cancelled) return;
      setSettings(s);
      setRay(s.ray);
      setCdsw(s.cdsw_workers);
    });
    return () => {
      cancelled = true;
    };
  }, [isProjectScope, projectId]);

  async function save() {
    if (!ray || !cdsw) return;
    setSaving(true);
    setMessage(null);
    try {
      const updated = isProjectScope
        ? await api.updateProjectRaySettings(projectId, { ray, cdsw_workers: cdsw })
        : await api.updatePlatformSettings({ ray, cdsw_workers: cdsw });
      setSettings(updated);
      setRay(updated.ray);
      setCdsw(updated.cdsw_workers);
      setMessage("Saved.");
      onSaved?.();
    } catch (e) {
      setMessage(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  }

  if (!settings || !ray || !cdsw) {
    return (
      <article className="card wide">
        <h2>{title}</h2>
        <p>{isProjectScope ? "Loading project Ray settings…" : "Loading platform settings…"}</p>
      </article>
    );
  }

  return (
    <article className="card wide">
      <h2>{title}</h2>
      <p className="hint">
        {introHint ??
          (isProjectScope
            ? "This project’s Ray cluster and CDSW worker targets apply to every task and job run under it."
            : "Defaults copied into new projects. Each project keeps its own Ray configuration for runs.")}{" "}
        Runtime: Ray <strong>{settings.runtime.ray_runtime}</strong>
        · CDSW API <strong>{settings.runtime.cdsw_available ? "available" : "local sim"}</strong>
        · detected head <code>{settings.runtime.detected_ray_head ?? "—"}</code>
      </p>

      <div className="settings-grid">
        <div>
          <h3>Ray</h3>
          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={ray.ray_local_mode}
              onChange={(e) => setRay({ ...ray, ray_local_mode: e.target.checked })}
            />
            Ray local mode (in-process CPU when not using remote workers)
          </label>
          <label>
            Ray head address (override)
            <input
              value={ray.ray_head_address ?? ""}
              placeholder="e.g. 100.100.252.18:6379"
              onChange={(e) =>
                setRay({
                  ...ray,
                  ray_head_address: e.target.value.trim() || null,
                })
              }
            />
          </label>
          <label>
            Ray Train worker cap (0 = auto)
            <input
              type="number"
              min={0}
              max={32}
              value={ray.ray_train_num_workers}
              onChange={(e) =>
                setRay({ ...ray, ray_train_num_workers: Number(e.target.value) })
              }
            />
          </label>
          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={ray.non_distributed_low_memory_mode}
              onChange={(e) =>
                setRay({
                  ...ray,
                  non_distributed_low_memory_mode: e.target.checked,
                })
              }
            />
            Non-distributed low memory mode (1 Train rank on session, no CDSW engines)
          </label>
          <p className="hint settings-hint-inline">
            Leave off for normal distributed jobs — worker engines supply the large memory pool;
            the session only runs Ray control. Turn on only for small session-only training.
          </p>
        </div>

        <div>
          <h3>CDSW workers</h3>
          <WorkerSizePresetBar
            selectedPresetId={workerPresetId}
            disabled={saving}
            onApply={applyWorkerPreset}
          />
          <label>
            Worker CPU
            <input
              type="number"
              min={0.1}
              step={0.5}
              value={cdsw.worker_cpu}
              onChange={(e) => {
                setWorkerPresetId(null);
                setCdsw({ ...cdsw, worker_cpu: Number(e.target.value) });
              }}
            />
          </label>
          <label>
            Worker memory (GiB)
            <input
              type="number"
              min={1}
              value={cdsw.worker_memory_gb}
              onChange={(e) => {
                setWorkerPresetId(null);
                setCdsw({ ...cdsw, worker_memory_gb: Number(e.target.value) });
              }}
            />
          </label>
          <label>
            GPUs per worker (default for new jobs)
            <input
              type="number"
              min={0}
              max={8}
              value={cdsw.gpus_per_worker}
              onChange={(e) => {
                setWorkerPresetId(null);
                setCdsw({ ...cdsw, gpus_per_worker: Number(e.target.value) });
              }}
            />
          </label>
          <label>
            Default job worker count
            <input
              type="number"
              min={1}
              max={32}
              value={cdsw.default_job_num_workers}
              onChange={(e) =>
                setCdsw({ ...cdsw, default_job_num_workers: Number(e.target.value) })
              }
            />
          </label>
          <label>
            CML accelerator label ID
            <input
              value={cdsw.accelerator_label_id ?? ""}
              onChange={(e) =>
                setCdsw({
                  ...cdsw,
                  accelerator_label_id: e.target.value.trim() || null,
                })
              }
            />
          </label>
          <label>
            CML resource profile ID
            <input
              value={cdsw.resource_profile_id ?? ""}
              onChange={(e) =>
                setCdsw({
                  ...cdsw,
                  resource_profile_id: e.target.value.trim() || null,
                })
              }
            />
          </label>
          <label>
            CML resource group ID
            <input
              value={cdsw.resource_group_id ?? ""}
              onChange={(e) =>
                setCdsw({
                  ...cdsw,
                  resource_group_id: e.target.value.trim() || null,
                })
              }
            />
          </label>
        </div>
      </div>

      <button type="button" onClick={save} disabled={saving}>
        {saving
          ? "Saving…"
          : saveLabel ?? (isProjectScope ? "Save project Ray settings" : "Save platform defaults")}
      </button>
      {message && <p className="hint">{message}</p>}
    </article>
  );
}
