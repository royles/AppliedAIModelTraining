import { useCallback, useEffect, useMemo, useState } from "react";
import { api, type RegistryLifecycleInfo, type RegistryModel } from "./api";
import { ModelRegistryChat } from "./ModelRegistryChat";

const STATE_CLASS: Record<string, string> = {
  trained: "trained",
  testing: "testing",
  benchmarked: "benchmarked",
  production: "production",
  retired: "retired",
};

type ProjectGroup = {
  key: string;
  projectId: number | null;
  projectName: string;
  models: RegistryModel[];
};

function groupModelsByProject(models: RegistryModel[]): ProjectGroup[] {
  const map = new Map<string, ProjectGroup>();
  for (const m of models) {
    const projectId = m.experiment_id;
    const projectName = m.project_name?.trim() || "Unassigned project";
    const key = projectId != null ? `project-${projectId}` : "unassigned";
    let group = map.get(key);
    if (!group) {
      group = { key, projectId, projectName, models: [] };
      map.set(key, group);
    }
    group.models.push(m);
  }
  const groups = Array.from(map.values());
  groups.sort((a, b) => a.projectName.localeCompare(b.projectName));
  for (const g of groups) {
    g.models.sort((a, b) => b.created_at_unix - a.created_at_unix);
  }
  return groups;
}

export function ModelRegistryPanel() {
  const [models, setModels] = useState<RegistryModel[]>([]);
  const [lifecycle, setLifecycle] = useState<RegistryLifecycleInfo[]>([]);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const selected = models.find((m) => m.id === selectedId) ?? null;
  const projectGroups = useMemo(() => groupModelsByProject(models), [models]);
  const lineage = (selected?.metadata?.lineage_snapshot ?? null) as Record<string, unknown> | null;
  const rayConfig = (lineage?.ray_config ?? null) as Record<string, unknown> | null;
  const jobRun = (lineage?.job_run ?? null) as Record<string, unknown> | null;
  const raySnap = (rayConfig?.ray ?? null) as Record<string, unknown> | null;
  const cdswSnap = (rayConfig?.cdsw_workers ?? null) as Record<string, unknown> | null;
  const launchSpec = (jobRun?.effective_worker_launch_spec ?? null) as Record<string, unknown> | null;

  const load = useCallback(async () => {
    try {
      const [m, lc] = await Promise.all([api.listRegistryModels(), api.registryLifecycle()]);
      setModels(m);
      setLifecycle(lc);
      setError(null);
      setSelectedId((prev) => (prev === null && m.length > 0 ? m[0].id : prev));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    load();
    const id = window.setInterval(load, 8000);
    return () => window.clearInterval(id);
  }, [load]);

  async function transition(state: string) {
    if (!selected) return;
    setBusy(true);
    try {
      const updated = await api.transitionRegistryModel(selected.id, state);
      setModels((prev) => prev.map((m) => (m.id === updated.id ? updated : m)));
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  const allowed = lifecycle.find((l) => l.state === selected?.lifecycle_state)?.transitions_to ?? [];

  return (
    <div className="registry-layout">
      <header className="main-panel-header">
        <h2>Model registry</h2>
        <p className="hint">
          Registered models are <strong>immutable</strong> snapshots (lineage, model card). Lifecycle
          changes only; new training requires a new job run and registry entry.
        </p>
      </header>

      {error && <div className="banner error">{error}</div>}

      <div className="registry-split">
        <aside className="registry-list card">
          <h3>Projects & models ({models.length})</h3>
          {models.length === 0 ? (
            <p className="hint">No registered models yet. Complete a training job in Model Tasks.</p>
          ) : (
            <div className="registry-project-groups">
              {projectGroups.map((group) => (
                <section key={group.key} className="registry-project-group">
                  <h4 className="registry-project-title">{group.projectName}</h4>
                  <ul className="registry-model-list">
                    {group.models.map((m) => (
                      <li key={m.id}>
                        <button
                          type="button"
                          className={selectedId === m.id ? "registry-item active" : "registry-item"}
                          onClick={() => setSelectedId(m.id)}
                        >
                          <span className="registry-item-title">
                            {m.display_name ?? m.openai_model_id}
                          </span>
                          <span className={`pill registry-state ${STATE_CLASS[m.lifecycle_state] ?? ""}`}>
                            {m.lifecycle_state}
                          </span>
                          <span className="hint registry-item-id">
                            {m.openai_model_id}
                            {m.job_name ? ` · job ${m.job_name}` : ""}
                          </span>
                        </button>
                      </li>
                    ))}
                  </ul>
                </section>
              ))}
            </div>
          )}
        </aside>

        {selected && (
          <section className="registry-detail card">
            <div className="registry-detail-head">
              <h3>{selected.display_name ?? selected.openai_model_id}</h3>
              <span className={`pill registry-state ${STATE_CLASS[selected.lifecycle_state] ?? ""}`}>
                {selected.lifecycle_state}
              </span>
            </div>
            <ul className="kv registry-meta">
              <li>
                <span>Project</span>
                <span>{selected.project_name ?? "—"}</span>
              </li>
              <li><span>OpenAI id</span><span><code>{selected.openai_model_id}</code></span></li>
              <li><span>Job run</span><span>#{selected.job_id} {selected.job_name ?? ""}</span></li>
              <li><span>Task type</span><span>{selected.task_type}</span></li>
              <li><span>Base</span><span>{selected.base_model}</span></li>
            </ul>

            {lineage && (
              <div className="registry-lineage-block" aria-label="Immutable lineage snapshot">
                <h4>Lineage (frozen at registration)</h4>
                <ul className="kv registry-meta">
                  <li>
                    <span>Task</span>
                    <span>
                      {String(lineage.task_name ?? "—")} v{String(lineage.task_version ?? "—")}
                    </span>
                  </li>
                  <li>
                    <span>Task version id</span>
                    <span>{String(lineage.task_version_id ?? "—")}</span>
                  </li>
                  <li>
                    <span>Config hash</span>
                    <span>
                      <code>{String(lineage.content_hash ?? "—").slice(0, 16)}…</code>
                    </span>
                  </li>
                </ul>
              </div>
            )}

            {rayConfig && (
              <div className="registry-lineage-block" aria-label="Ray configuration snapshot">
                <h4>Ray configuration (frozen)</h4>
                <ul className="kv registry-meta">
                  <li>
                    <span>Ray head</span>
                    <span>
                      <code>{String(raySnap?.ray_head_address ?? "auto / env")}</code>
                    </span>
                  </li>
                  <li>
                    <span>Train worker cap</span>
                    <span>{String(raySnap?.ray_train_num_workers ?? "—")}</span>
                  </li>
                  <li>
                    <span>Worker sizing</span>
                    <span>
                      {String(cdswSnap?.worker_cpu ?? "—")} CPU · {String(cdswSnap?.worker_memory_gb ?? "—")}{" "}
                      GiB · {String(cdswSnap?.gpus_per_worker ?? "—")} GPU/worker
                    </span>
                  </li>
                </ul>
              </div>
            )}

            {jobRun && (
              <div className="registry-lineage-block" aria-label="Job run snapshot">
                <h4>Job run (frozen)</h4>
                <ul className="kv registry-meta">
                  <li>
                    <span>Status</span>
                    <span>{String(jobRun.status ?? "—")}</span>
                  </li>
                  <li>
                    <span>Ray address</span>
                    <span>
                      <code>{String(jobRun.ray_address ?? "—")}</code>
                    </span>
                  </li>
                  <li>
                    <span>Workers</span>
                    <span>
                      {String(jobRun.num_workers ?? "—")} × {String(jobRun.gpus_per_worker ?? "—")} GPU
                    </span>
                  </li>
                  {launchSpec && (
                    <li>
                      <span>CDSW launch</span>
                      <span>
                        {String(launchSpec.cpu ?? "—")} CPU · {String(launchSpec.memory_gb ?? "—")} GiB
                      </span>
                    </li>
                  )}
                </ul>
              </div>
            )}

            {allowed.length > 0 && (
              <div className="registry-actions">
                <span className="hint">Transition:</span>
                {allowed.map((st) => (
                  <button
                    key={st}
                    type="button"
                    className="btn-secondary"
                    disabled={busy}
                    onClick={() => transition(st)}
                  >
                    → {st}
                  </button>
                ))}
              </div>
            )}

            <details className="registry-lifecycle-help">
              <summary>Lifecycle guidance</summary>
              <ul className="registry-lifecycle-list">
                {lifecycle.map((l) => (
                  <li key={l.state}>
                    <strong>{l.state}</strong> — {l.guidance}
                  </li>
                ))}
              </ul>
            </details>

            <details className="registry-model-card-details">
              <summary>Model card (markdown)</summary>
              <pre className="model-card-md">{selected.model_card_markdown ?? "No card yet."}</pre>
            </details>

            <ModelRegistryChat model={selected} />
          </section>
        )}
      </div>
    </div>
  );
}
