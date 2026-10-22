import type { TrainingDatasetConfig } from "./api";
import type { WizardValidationMessage } from "./wizardValidationMessage";

export type DatasetValidationResult = {
  valid: boolean;
  block: WizardValidationMessage | null;
};

const HF_REPO_PATTERN = /^[\w.-]+\/[\w.-]+$/;

const HF_CONFIG_REQUIRED_RE = /^Salesforce\/wikitext$/i;

function invalid(message: string, fix: string): DatasetValidationResult {
  return { valid: false, block: { message, fix } };
}

function ok(): DatasetValidationResult {
  return { valid: true, block: null };
}

function parseHfDatasetId(raw: string): { repo: string; inlineConfig: string | null } {
  const text = raw.trim();
  for (const sep of [":", "@"] as const) {
    if (text.includes(sep)) {
      const [repo, config] = text.split(sep, 2);
      const r = repo?.trim() ?? "";
      const c = config?.trim() ?? "";
      if (r && c) {
        return { repo: r, inlineConfig: c };
      }
    }
  }
  return { repo: text, inlineConfig: null };
}

function validateHuggingFace(dataset: TrainingDatasetConfig): DatasetValidationResult {
  const id = dataset.hf_dataset?.trim();
  if (!id) {
    return invalid(
      "Hugging Face dataset id is missing.",
      'Set the dataset field to something like Salesforce/wikitext or Salesforce/wikitext:wikitext-2-raw-v1.',
    );
  }
  const { repo, inlineConfig } = parseHfDatasetId(id);
  if (!HF_REPO_PATTERN.test(repo)) {
    return invalid(
      "Dataset id format is not recognized.",
      "Use organization/dataset-name (letters, digits, dots, hyphens, underscores).",
    );
  }
  const config = (dataset.hf_config?.trim() || inlineConfig) ?? "";
  if (HF_CONFIG_REQUIRED_RE.test(repo) && !config) {
    return invalid(
      "WikiText needs a config subset.",
      'Add hf_config wikitext-2-raw-v1, or set the id to Salesforce/wikitext:wikitext-2-raw-v1. Data loads when the job runs.',
    );
  }
  if (!(dataset.hf_split?.trim())) {
    return invalid(
      "HF split is empty.",
      'Set split to train (or validation/test if you intend to use those).',
    );
  }
  if (!(dataset.hf_text_column?.trim())) {
    return invalid(
      "HF text column is empty.",
      'Set text column to text unless your dataset uses a different column name.',
    );
  }
  return ok();
}

/** Local checks only — no network or dataset download. */
export function validateDatasetConfig(dataset: TrainingDatasetConfig): DatasetValidationResult {
  switch (dataset.source) {
    case "builtin":
      return dataset.builtin_name?.trim()
        ? ok()
        : invalid("No built-in corpus selected.", "Pick platform demo or switch to another source.");
    case "inline":
      return (dataset.texts?.length ?? 0) > 0
        ? ok()
        : invalid(
            "No inline training text.",
            "Paste at least one example in the textarea (blank line between examples).",
          );
    case "text_file":
    case "jsonl_file": {
      const path = dataset.path?.trim();
      if (!path) {
        return invalid(
          "No dataset file path.",
          "Choose a file under data/datasets/ or type a relative path such as starter_corpus.txt.",
        );
      }
      if (path.includes("..")) {
        return invalid(
          "File path is not allowed.",
          "Use a path relative to data/datasets/ without .. segments.",
        );
      }
      return ok();
    }
    case "huggingface":
      return validateHuggingFace(dataset);
    case "iceberg":
      return dataset.iceberg_table?.trim()
        ? ok()
        : invalid(
            "Iceberg table is missing.",
            "Enter namespace.table and fill catalog URI/warehouse if your catalog needs them.",
          );
    case "parquet_s3": {
      const uri = dataset.parquet_uri?.trim();
      if (!uri) {
        return invalid(
          "S3 Parquet URI is missing.",
          "Use s3://your-bucket/prefix/ pointing at Parquet files.",
        );
      }
      if (!uri.startsWith("s3://")) {
        return invalid(
          "Parquet URI must be an S3 URL.",
          "Prefix the path with s3:// (credentials are resolved when the job runs).",
        );
      }
      return ok();
    }
    default:
      return invalid("Unknown dataset source.", "Pick a source from the dropdown.");
  }
}

/** @deprecated Prefer validateDatasetConfig().valid */
export function datasetConfigLooksComplete(dataset: TrainingDatasetConfig): boolean {
  return validateDatasetConfig(dataset).valid;
}
