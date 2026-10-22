import type { ReactNode } from "react";
import type { DatasetPreview } from "../trainingBlueprint";
import type { TrainingDatasetConfig } from "../api";
import { datasetConfigLooksComplete } from "../datasetValidation";

type Props = {
  dataset: TrainingDatasetConfig;
  preview: DatasetPreview | null;
  previewLoading: boolean;
  previewError: string | null;
  minCorpusChars: number | null;
};

export function DatasetPreviewBanner({
  dataset,
  preview,
  previewLoading,
  previewError,
  minCorpusChars,
}: Props) {
  if (dataset.source === "builtin") {
    return (
      <aside className="dataset-preview-banner dataset-preview-banner-muted" aria-live="polite">
        <strong>Dataset preview</strong>
        <p className="hint">Built-in demo corpus — no remote preview.</p>
      </aside>
    );
  }

  const configured = datasetConfigLooksComplete(dataset);

  let body: ReactNode;
  if (!configured) {
    body = (
      <p className="hint">Complete the dataset fields below; preview runs automatically after you pause typing.</p>
    );
  } else if (previewLoading) {
    body = (
      <p className="hint">
        <span className="dataset-preview-spinner" aria-hidden /> Loading preview (first ~100 HF rows only)…
      </p>
    );
  } else if (previewError) {
    body = <p className="dataset-preview-error">{previewError}</p>;
  } else if (preview) {
    const min = minCorpusChars ?? 10_000;
    const meetsMin = preview.total_characters >= min;
    body = (
      <>
        <p className="dataset-preview-stats">
          <span>
            <strong>{preview.examples.toLocaleString()}</strong> examples
          </span>
          <span>
            <strong>{preview.total_characters.toLocaleString()}</strong> characters
          </span>
          {preview.preview_capped && (
            <span className="dataset-preview-cap">
              Summary of first {preview.sample_limit ?? 100} records — training uses the full split unless{" "}
              <code>max_samples</code> is set.
            </span>
          )}
        </p>
        {minCorpusChars != null && (
          <p className={`hint ${meetsMin ? "dataset-preview-ok" : "blueprint-warn"}`}>
            BPE minimum for this vocab: {min.toLocaleString()} characters
            {meetsMin ? " — preview meets minimum." : " — add more text or lower vocab size."}
          </p>
        )}
      </>
    );
  } else {
    body = <p className="hint">Waiting to run preview…</p>;
  }

  return (
    <aside className="dataset-preview-banner" aria-live="polite" aria-label="Dataset preview">
      <strong>Dataset preview</strong>
      {body}
    </aside>
  );
}
