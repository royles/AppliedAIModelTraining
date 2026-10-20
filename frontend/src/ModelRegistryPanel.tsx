import { useCallback, useEffect, useState } from "react";
import { api, type RegistryLifecycleInfo, type RegistryModel } from "./api";
import { ModelRegistryChat } from "./ModelRegistryChat";

const STATE_CLASS: Record<string, string> = {
  trained: "trained",
  testing: "testing",
  benchmarked: "benchmarked",
  production: "production",
  retired: "retired",
};

export function ModelRegistryPanel() {
  const [models, setModels] = useState<RegistryModel[]>([]);
  const [lifecycle, setLifecycle] = useState<RegistryLifecycleInfo[]>([]);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const selected = models.find((m) => m.id === selectedId) ?? null;

  const load = useCallback(async () => {
    try {
      const [m, lc] = await Promise.all([api.listRegistryModels(), api.registryLifecycle()]);
      setModels(m);
      setLifecycle(lc);
      setError(null);
      if (selectedId === null && m.length > 0) {
        setSelectedId(m[0].id);
      }
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
          Trained models are registered automatically with metadata and a model card. Manage
          create → train → test → benchmark → production → retire here.
        </p>
      </header>

      {error && <div className="banner error">{error}</div>}

      <div className="registry-split">
        <aside className="registry-list card">
          <h3>Models ({models.length})</h3>
          {models.length === 0 ? (
            <p className="hint">No registered models yet. Complete a causal-LM training task.</p>
          ) : (
            <ul className="registry-model-list">
              {models.map((m) => (
                <li key={m.id}>
                  <button
                    type="button"
                    className={selectedId === m.id ? "registry-item active" : "registry-item"}
                    onClick={() => setSelectedId(m.id)}
                  >
                    <span className="registry-item-title">{m.display_name ?? m.openai_model_id}</span>
                    <span className={`pill registry-state ${STATE_CLASS[m.lifecycle_state] ?? ""}`}>
                      {m.lifecycle_state}
                    </span>
                    <span className="hint registry-item-id">{m.openai_model_id}</span>
                  </button>
                </li>
              ))}
            </ul>
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
              <li><span>OpenAI id</span><span><code>{selected.openai_model_id}</code></span></li>
              <li><span>Training job</span><span>#{selected.job_id} {selected.job_name ?? ""}</span></li>
              <li><span>Task</span><span>{selected.task_type}</span></li>
              <li><span>Base</span><span>{selected.base_model}</span></li>
            </ul>

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

            <h4>Model card</h4>
            <pre className="model-card-md">{selected.model_card_markdown ?? "No card yet."}</pre>

            <ModelRegistryChat model={selected} />
          </section>
        )}
      </div>
    </div>
  );
}
