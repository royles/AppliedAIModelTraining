import { useEffect, useRef, useState } from "react";
import { api, type TrainingDatasetConfig } from "../api";
import type { DatasetPreview } from "../trainingBlueprint";

const DEFAULT_DEBOUNCE_MS = 650;

export function useDebouncedDatasetPreview(
  dataset: TrainingDatasetConfig,
  debounceMs: number = DEFAULT_DEBOUNCE_MS,
) {
  const [preview, setPreview] = useState<DatasetPreview | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const requestIdRef = useRef(0);

  useEffect(() => {
    if (dataset.source === "builtin") {
      setPreviewLoading(false);
      setPreviewError(null);
      setPreview(null);
      return;
    }

    const id = ++requestIdRef.current;
    setPreviewLoading(true);
    setPreviewError(null);

    const timer = window.setTimeout(() => {
      api
        .datasetPreview(dataset)
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
    };
  }, [dataset, debounceMs]);

  return { preview, previewError, previewLoading };
}
