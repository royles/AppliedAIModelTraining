import type { TrainingDatasetConfig } from "./api";

/** Local checks only — no network; used so the wizard can advance while preview runs. */
export function datasetConfigLooksComplete(dataset: TrainingDatasetConfig): boolean {
  switch (dataset.source) {
    case "builtin":
      return Boolean(dataset.builtin_name?.trim());
    case "inline":
      return (dataset.texts?.length ?? 0) > 0;
    case "text_file":
    case "jsonl_file":
      return Boolean(dataset.path?.trim());
    case "huggingface":
      return Boolean(dataset.hf_dataset?.trim());
    case "iceberg":
      return Boolean(dataset.iceberg_table?.trim());
    case "parquet_s3":
      return Boolean(dataset.parquet_uri?.trim());
    default:
      return false;
  }
}
