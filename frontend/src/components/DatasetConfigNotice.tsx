import type { TrainingDatasetConfig } from "../api";

export function DatasetConfigNotice({ dataset }: { dataset: TrainingDatasetConfig }) {
  if (dataset.source === "builtin") {
    return null;
  }
  return (
    <aside className="dataset-preview-banner dataset-preview-banner-muted" aria-label="Dataset">
      <strong>Dataset</strong>
      <p className="hint">
        Loading, tokenization, and corpus checks run when you <strong>Start job</strong>, not while
        editing this wizard.
      </p>
    </aside>
  );
}
