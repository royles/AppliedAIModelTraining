import { useEffect, useState } from "react";
import { api, type TrainingDatasetConfig } from "./api";

type Props = {
  value: TrainingDatasetConfig;
  onChange: (next: TrainingDatasetConfig) => void;
};

export function DatasetConfigFields({ value, onChange }: Props) {
  const [files, setFiles] = useState<{ path: string; kind: string }[]>([]);

  useEffect(() => {
    api.listDatasetFiles().then(setFiles).catch(() => setFiles([]));
  }, []);

  function patch(partial: Partial<TrainingDatasetConfig>) {
    onChange({ ...value, ...partial });
  }

  return (
    <article className="card">
      <h3>Training dataset</h3>
      <p className="hint">
        Corpus used during Ray Train. Files live under <code>data/datasets/</code> in the project.
      </p>
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
            rows={8}
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

      {value.source === "huggingface" && (
        <>
          <label>
            Dataset id
            <input
              value={value.hf_dataset ?? ""}
              onChange={(e) => patch({ hf_dataset: e.target.value.trim() || null })}
              placeholder="wikitext"
            />
          </label>
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
          <input
            type="number"
            min={1}
            max={10000}
            value={value.repeat ?? 1}
            onChange={(e) => patch({ repeat: Number(e.target.value) })}
          />
        </label>
        <label>
          Max samples (optional)
          <input
            type="number"
            min={1}
            value={value.max_samples ?? ""}
            onChange={(e) =>
              patch({
                max_samples: e.target.value ? Number(e.target.value) : null,
              })
            }
          />
        </label>
      </div>
    </article>
  );
}
