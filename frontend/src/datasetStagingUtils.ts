import type { TrainingDatasetConfig } from "./api";

/** Iceberg and S3 Parquet — read at training time, no local corpus replication. */
export function isRemoteDatasetSource(
  source: TrainingDatasetConfig["source"],
): boolean {
  return source === "iceberg" || source === "parquet_s3";
}
