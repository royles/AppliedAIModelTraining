import { useEffect, useState } from "react";
import {
  api,
  type CdswWorkerSettings,
  type PlatformSettings,
  type RaySettings,
} from "./api";

type Props = {
  onSaved?: () => void;
};

export function PlatformSettingsPanel({ onSaved }: Props) {
  const [settings, setSettings] = useState<PlatformSettings | null>(null);
  const [ray, setRay] = useState<RaySettings | null>(null);
  const [cdsw, setCdsw] = useState<CdswWorkerSettings | null>(null);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  useEffect(() => {
    api.getPlatformSettings().then((s) => {
      setSettings(s);
      setRay(s.ray);
      setCdsw(s.cdsw_workers);
    });
  }, []);

  async function save() {
    if (!ray || !cdsw) return;
    setSaving(true);
    setMessage(null);
    try {
      const updated = await api.updatePlatformSettings({ ray, cdsw_workers: cdsw });
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
        <h2>Ray &amp; CDSW workers</h2>
        <p>Loading platform settings…</p>
      </article>
    );
  }

  return (
    <article className="card wide">
      <h2>Ray &amp; CDSW workers</h2>
      <p className="hint">
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
        </div>

        <div>
          <h3>CDSW workers</h3>
          <label>
            Worker CPU
            <input
              type="number"
              min={0.1}
              step={0.5}
              value={cdsw.worker_cpu}
              onChange={(e) => setCdsw({ ...cdsw, worker_cpu: Number(e.target.value) })}
            />
          </label>
          <label>
            Worker memory (GiB)
            <input
              type="number"
              min={1}
              value={cdsw.worker_memory_gb}
              onChange={(e) =>
                setCdsw({ ...cdsw, worker_memory_gb: Number(e.target.value) })
              }
            />
          </label>
          <label>
            GPUs per worker (default for new jobs)
            <input
              type="number"
              min={0}
              max={8}
              value={cdsw.gpus_per_worker}
              onChange={(e) =>
                setCdsw({ ...cdsw, gpus_per_worker: Number(e.target.value) })
              }
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
        {saving ? "Saving…" : "Save platform settings"}
      </button>
      {message && <p className="hint">{message}</p>}
    </article>
  );
}
