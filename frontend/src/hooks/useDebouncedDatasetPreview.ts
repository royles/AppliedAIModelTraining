import { useEffect, useMemo, useRef, useState } from "react";
import { api, type TrainingDatasetConfig } from "../api";
import { datasetConfigLooksComplete } from "../datasetValidation";
import type { DatasetPreview } from "../trainingBlueprint";

const DEFAULT_DEBOUNCE_MS = 650;

export function useDebouncedDatasetPreview(
  dataset: TrainingDatasetConfig,
  debounceMs: number = DEFAULT_DEBOUNCE_MS,
  taskType: "causal_lm" | "sequence_classification" = "causal_lm",
) {
  const [preview, setPreview] = useState<DatasetPreview | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const requestIdRef = useRef(0);
  const datasetKey = useMemo(() => JSON.stringify(dataset), [dataset]);

  useEffect(() => {
    const cfg = JSON.parse(datasetKey) as TrainingDatasetConfig;
    if (cfg.source === "builtin") {
      setPreviewLoading(false);
      setPreviewError(null);
      setPreview(null);
      return;
    }
    if (!datasetConfigLooksComplete(cfg)) {
      setPreviewLoading(false);
      setPreviewError(null);
      setPreview(null);
      return;
    }

    const id = ++requestIdRef.current;
    let fetchStarted = false;

    const timer = window.setTimeout(() => {
      fetchStarted = true;
      setPreviewLoading(true);
      setPreviewError(null);
      api
        .datasetPreview(cfg, taskType)
        .then((p) => {
          if (requestIdRef.current !== id) return;
          setPreview(p);
          setPreviewError(null);
        })
        .catch((e) => {
          if (requestIdRef.current !== id) return;
          setPreview(null);
          setPreviewError(e instanceof Error ? e.message : String(e));
        })
        .finally(() => {
          if (requestIdRef.current === id) {
            setPreviewLoading(false);
          }
        });
    }, debounceMs);

    return () => {
      window.clearTimeout(timer);
      if (!fetchStarted && requestIdRef.current === id) {
        setPreviewLoading(false);
      }
    };
  }, [datasetKey, debounceMs, taskType]);

  return { preview, previewError, previewLoading };
}
