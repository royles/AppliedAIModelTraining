import { useEffect, useState } from "react";
import { api, type TrainingDatasetConfig } from "./api";
import { ValidatedNumberField } from "./ValidatedNumberField";

type Props = {
  value: TrainingDatasetConfig;
  onChange: (next: TrainingDatasetConfig) => void;
  compact?: boolean;
};

export function DatasetConfigFields({ value, onChange, compact = false }: Props) {
  const [files, setFiles] = useState<{ path: string; kind: string }[]>([]);

  useEffect(() => {
    api.listDatasetFiles().then(setFiles).catch(() => setFiles([]));
  }, []);

  function patch(partial: Partial<TrainingDatasetConfig>) {
    onChange({ ...value, ...partial });
  }

  const Wrapper = compact ? "div" : "article";
  return (
    <Wrapper className={compact ? "wizard-step-compact dataset-fields-compact" : "card"}>
      {!compact && <h3>Training dataset</h3>}
      {!compact && (
        <p className="hint">
          Corpus for Ray Train: project files, Hugging Face, Iceberg tables, or Parquet on S3
          (<code>s3://bucket/prefix/</code>). Session needs S3/Iceberg credentials via env or CAI mounts.
        </p>
      )}
      <label>
        Source
        <select
          value={value.source}
          onChange={(e) =>
            patch({ source: e.target.value as TrainingDatasetConfig["source"] })
          }
        >
          <option value="builtin">Built-in demo</option>
          <option value="inline">Paste examples</option>
          <option value="text_file">Text file (.txt)</option>
          <option value="jsonl_file">JSONL file</option>
          <option value="huggingface">Hugging Face dataset</option>
          <option value="iceberg">Iceberg table</option>
          <option value="parquet_s3">Parquet on S3</option>
        </select>
      </label>

      {value.source === "builtin" && (
        <label>
          Built-in preset
          <select
            value={value.builtin_name ?? "platform_demo"}
            onChange={(e) => patch({ builtin_name: e.target.value })}
          >
            <option value="platform_demo">Platform demo (Ray / CDSW)</option>
          </select>
        </label>
      )}

      {value.source === "inline" && (
        <label>
          Training texts (one example per blank line)
          <textarea
            rows={compact ? 3 : 8}
            value={(value.texts ?? []).join("\n\n")}
            onChange={(e) => {
              const texts = e.target.value
                .split(/\n\s*\n/)
                .map((t) => t.trim())
                .filter(Boolean);
              patch({ texts });
            }}
            placeholder={"User: Hello\nAssistant: Hi there!"}
          />
        </label>
      )}

      {(value.source === "text_file" || value.source === "jsonl_file") && (
        <>
          <label>
            File path (under data/datasets/)
            <select
              value={value.path ?? ""}
              onChange={(e) => patch({ path: e.target.value || null })}
            >
              <option value="">Select a file…</option>
              {files
                .filter((f) =>
                  value.source === "jsonl_file" ? f.kind === "jsonl_file" : f.kind === "text_file",
                )
                .map((f) => (
                  <option key={f.path} value={f.path}>{f.path}</option>
                ))}
            </select>
          </label>
          <label>
            Or type path
            <input
              value={value.path ?? ""}
              onChange={(e) => patch({ path: e.target.value.trim() || null })}
              placeholder="example_causal.jsonl"
            />
          </label>
          {value.source === "jsonl_file" && (
            <label>
              Text column
              <input
                value={value.text_field ?? "text"}
                onChange={(e) => patch({ text_field: e.target.value })}
              />
            </label>
          )}
        </>
      )}

      {value.source === "iceberg" && (
        <>
          <label>
            Catalog URI (REST)
            <input
              value={value.iceberg_catalog_uri ?? ""}
              onChange={(e) => patch({ iceberg_catalog_uri: e.target.value.trim() || null })}
              placeholder="https://iceberg-catalog.example.com"
            />
          </label>
          <p className="hint">
            Or set <code>PLATFORM_ICEBERG_CATALOG_URI</code> on the session. Warehouse path optional
            for Hive-style catalogs.
          </p>
          <label>
            Table (namespace.table)
            <input
              value={value.iceberg_table ?? ""}
              onChange={(e) => patch({ iceberg_table: e.target.value.trim() || null })}
              placeholder="ml.training_corpus"
            />
          </label>
          <label>
            Text column
            <input
              value={value.iceberg_text_column ?? "text"}
              onChange={(e) => patch({ iceberg_text_column: e.target.value })}
            />
          </label>
          <label>
            Label column (optional)
            <input
              value={value.iceberg_label_column ?? ""}
              onChange={(e) =>
                patch({ iceberg_label_column: e.target.value.trim() || null })
              }
            />
          </label>
          <label>
            Warehouse (optional)
            <input
              value={value.iceberg_warehouse ?? ""}
              onChange={(e) => patch({ iceberg_warehouse: e.target.value.trim() || null })}
              placeholder="s3://warehouse-bucket/iceberg/"
            />
          </label>
        </>
      )}

      {value.source === "parquet_s3" && (
        <>
          <label>
            Parquet location
            <input
              value={value.parquet_uri ?? ""}
              onChange={(e) => patch({ parquet_uri: e.target.value.trim() || null })}
              placeholder="s3://my-bucket/datasets/train/"
            />
          </label>
          <p className="hint">
            Single <code>.parquet</code> file or directory of Parquet parts. Uses{" "}
            <code>AWS_REGION</code> / instance credentials unless you set region below.
          </p>
          <label>
            Text column
            <input
              value={value.parquet_text_column ?? "text"}
              onChange={(e) => patch({ parquet_text_column: e.target.value })}
            />
          </label>
          <label>
            Label column (optional)
            <input
              value={value.parquet_label_column ?? ""}
              onChange={(e) =>
                patch({ parquet_label_column: e.target.value.trim() || null })
              }
            />
          </label>
          <label>
            S3 region (optional)
            <input
              value={value.s3_region ?? ""}
              onChange={(e) => patch({ s3_region: e.target.value.trim() || null })}
              placeholder="us-east-1"
            />
          </label>
        </>
      )}

      {value.source === "huggingface" && (
        <>
          <label>
            Dataset id
            <input
              value={value.hf_dataset ?? ""}
              onChange={(e) => patch({ hf_dataset: e.target.value.trim() || null })}
              placeholder="Salesforce/wikitext"
            />
          </label>
          <label>
            Config / subset (if required)
            <input
              value={value.hf_config ?? ""}
              onChange={(e) => patch({ hf_config: e.target.value.trim() || null })}
              placeholder="wikitext-103-raw-v1"
            />
          </label>
          <p className="hint">
            Some hubs need a config (WikiText: <code>wikitext-103-raw-v1</code>). You can also
            use <code>Salesforce/wikitext:wikitext-103-raw-v1</code> in the dataset id field.
            Preview uses a quick ~100-record summary (non-blocking). Training loads the full split
            unless you set <strong>Max samples</strong>.
          </p>
          <label>
            Split
            <input
              value={value.hf_split ?? "train"}
              onChange={(e) => patch({ hf_split: e.target.value })}
            />
          </label>
          <label>
            Text column
            <input
              value={value.hf_text_column ?? "text"}
              onChange={(e) => patch({ hf_text_column: e.target.value })}
            />
          </label>
        </>
      )}

      <div className="arch-grid arch-grid-compact">
        <label>
          Repeat corpus
          <ValidatedNumberField
            value={value.repeat ?? 1}
            min={1}
            max={10_000}
            onChange={(n) => patch({ repeat: n })}
          />
        </label>
        <label>
          Max samples (optional)
          <input
            type="text"
            inputMode="numeric"
            value={value.max_samples ?? ""}
            placeholder="All examples"
            onChange={(e) => {
              const t = e.target.value.trim();
              if (!t) {
                patch({ max_samples: null });
                return;
              }
              const n = Number.parseInt(t, 10);
              if (Number.isFinite(n) && n >= 1) {
                patch({ max_samples: n });
              }
            }}
          />
        </label>
      </div>
    </Wrapper>
  );
}
