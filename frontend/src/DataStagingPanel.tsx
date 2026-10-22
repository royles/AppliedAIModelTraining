import { useCallback, useEffect, useMemo, useState } from "react";
import { api, type TrainingDatasetConfig } from "./api";
import { DatasetConfigFields } from "./DatasetConfigFields";
import { DEFAULT_DATASET } from "./modelTaskDefaults";
import { ValidatedNumberField } from "./ValidatedNumberField";
import { isRemoteDatasetSource } from "./datasetStagingUtils";

type CollectionRow = {
  collection_id: string;
  name: string;
  description?: string | null;
  local_path: string;
  last_staged?: Record<string, unknown> | null;
};

type StageResult = {
  access_mode: "local" | "remote";
  local_path: string;
  corpus_local_path?: string | null;
  cache_local_path?: string;
  remote_uri?: string | null;
  corpus_examples?: number | null;
  last_staged?: Record<string, unknown>;
};

export function DataStagingPanel() {
  const [collections, setCollections] = useState<CollectionRow[]>([]);
  const [selectedId, setSelectedId] = useState<string>("");
  const [newName, setNewName] = useState("");
  const [newDescription, setNewDescription] = useState("");
  const [dataset, setDataset] = useState<TrainingDatasetConfig>(DEFAULT_DATASET);
  const [taskType, setTaskType] = useState<"causal_lm" | "sequence_classification">("causal_lm");
  const [rayWorkers, setRayWorkers] = useState(4);
  const [preShard, setPreShard] = useState(false);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [lastResult, setLastResult] = useState<StageResult | null>(null);
  const [detail, setDetail] = useState<Record<string, unknown> | null>(null);

  const remoteSource = isRemoteDatasetSource(dataset.source);

  const refresh = useCallback(async () => {
    const rows = await api.listDataCollections();
    setCollections(rows);
    if (!selectedId && rows.length > 0) {
      setSelectedId(rows[0].collection_id);
    }
  }, [selectedId]);

  useEffect(() => {
    refresh().catch((e) => setMessage(e instanceof Error ? e.message : String(e)));
  }, [refresh]);

  useEffect(() => {
    if (!selectedId) {
      setDetail(null);
      return;
    }
    api
      .getDataCollection(selectedId)
      .then((d) => setDetail(d))
      .catch(() => setDetail(null));
  }, [selectedId, collections]);

  const selected = useMemo(
    () => collections.find((c) => c.collection_id === selectedId) ?? null,
    [collections, selectedId],
  );

  async function handleCreateCollection() {
    const name = newName.trim();
    if (!name) {
      setMessage("Enter a collection name.");
      return;
    }
    setBusy(true);
    setMessage(null);
    try {
      const created = await api.createDataCollection({
        name,
        description: newDescription.trim() || undefined,
      });
      setNewName("");
      setNewDescription("");
      await refresh();
      setSelectedId(created.collection_id);
      setMessage(`Collection created. Registry path: ${created.local_path}`);
    } catch (e) {
      setMessage(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function handleStage() {
    if (!selectedId) {
      setMessage("Create or select a data collection first.");
      return;
    }
    setBusy(true);
    setMessage(null);
    try {
      const result = await api.stageDataCollection(selectedId, {
        dataset,
        task_type: taskType,
        ray_train_workers: remoteSource || !preShard ? undefined : rayWorkers,
      });
      setLastResult(result);
      await refresh();
      if (result.access_mode === "remote") {
        setMessage(
          `Registered remote dataset. Training reads from ${result.remote_uri ?? "remote storage"}. ` +
            `Collection registry: ${result.local_path}`,
        );
      } else {
        setMessage(
          `Corpus materialized locally (${result.corpus_examples ?? "?"} examples). ` +
            `Corpus path: ${result.corpus_local_path ?? result.local_path}`,
        );
      }
    } catch (e) {
      setMessage(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="data-staging-page">
      <header className="main-panel-header">
        <h2>Data</h2>
        <p className="hint">
          Shared data collections can be reused across many training runs. Local sources (files,
          Hugging Face, inline text) can be pre-loaded into a collection directory.{" "}
          <strong>Iceberg and S3 Parquet</strong> stay remote — register the collection here; runs
          read from the catalog or bucket without copying the corpus locally.
        </p>
      </header>

      {message && <div className="banner info">{message}</div>}

      <div className="data-staging-layout">
        <section className="card data-staging-collections" aria-labelledby="data-collections-heading">
          <h3 id="data-collections-heading">Shared collections</h3>
          <label>
            Active collection
            <select
              value={selectedId}
              onChange={(e) => setSelectedId(e.target.value)}
              disabled={collections.length === 0}
            >
              {collections.length === 0 && <option value="">— none yet —</option>}
              {collections.map((c) => (
                <option key={c.collection_id} value={c.collection_id}>
                  {c.name}
                </option>
              ))}
            </select>
          </label>
          {selected && (
            <dl className="data-staging-paths">
              <div>
                <dt>Registry local path</dt>
                <dd>
                  <code>{selected.local_path}</code>
                </dd>
              </div>
            </dl>
          )}

          <fieldset className="data-staging-new">
            <legend>New collection</legend>
            <label>
              Name
              <input
                type="text"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                placeholder="e.g. Wiki corpus"
              />
            </label>
            <label>
              Description
              <input
                type="text"
                value={newDescription}
                onChange={(e) => setNewDescription(e.target.value)}
                placeholder="Optional"
              />
            </label>
            <button type="button" className="btn-secondary" disabled={busy} onClick={handleCreateCollection}>
              Create collection
            </button>
          </fieldset>
        </section>

        <section className="card data-staging-stage" aria-labelledby="data-stage-heading">
          <h3 id="data-stage-heading">Pre-load / register</h3>
          <label>
            Task type
            <select
              value={taskType}
              onChange={(e) =>
                setTaskType(e.target.value as "causal_lm" | "sequence_classification")
              }
            >
              <option value="causal_lm">Causal LM</option>
              <option value="sequence_classification">Sequence classification</option>
            </select>
          </label>

          <DatasetConfigFields value={dataset} onChange={setDataset} compact />

          {remoteSource ? (
            <p className="hint data-staging-remote-note">
              This source is read from remote storage at training time. Staging only saves the
              collection binding and paths below — it does not download the dataset.
            </p>
          ) : (
            <label className="data-staging-shard-toggle">
              <input
                type="checkbox"
                checked={preShard}
                onChange={(e) => setPreShard(e.target.checked)}
              />
              Pre-shard for Ray Train workers (optional)
            </label>
          )}

          {!remoteSource && preShard && (
            <label>
              Ray worker count
              <ValidatedNumberField value={rayWorkers} min={2} max={128} onChange={setRayWorkers} />
            </label>
          )}

          <button type="button" className="btn-primary" disabled={busy || !selectedId} onClick={handleStage}>
            {remoteSource ? "Register remote dataset" : "Stage corpus locally"}
          </button>

          {lastResult && (
            <dl className="data-staging-paths data-staging-result">
              <div>
                <dt>Access</dt>
                <dd>{lastResult.access_mode === "remote" ? "Remote (no local copy)" : "Local corpus"}</dd>
              </div>
              <div>
                <dt>Collection local path</dt>
                <dd>
                  <code>{lastResult.local_path}</code>
                </dd>
              </div>
              {lastResult.access_mode === "remote" ? (
                <div>
                  <dt>Remote URI</dt>
                  <dd>
                    <code>{lastResult.remote_uri ?? "—"}</code>
                  </dd>
                </div>
              ) : (
                <>
                  <div>
                    <dt>Corpus local path</dt>
                    <dd>
                      <code>{lastResult.corpus_local_path ?? "—"}</code>
                    </dd>
                  </div>
                  {lastResult.cache_local_path && (
                    <div>
                      <dt>Shard cache local path</dt>
                      <dd>
                        <code>{lastResult.cache_local_path}</code>
                      </dd>
                    </div>
                  )}
                </>
              )}
            </dl>
          )}

          {detail?.last_staged && typeof detail.last_staged === "object" && (
            <p className="hint">
              Last registered:{" "}
              {String((detail.last_staged as Record<string, unknown>).staged_at ?? "")}
            </p>
          )}
        </section>
      </div>
    </div>
  );
}
