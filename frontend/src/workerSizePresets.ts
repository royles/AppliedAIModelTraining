/** CDSW/CML worker sizing presets (must fit your CML resource profile). */

export type WorkerSizePreset = {
  id: string;
  label: string;
  worker_cpu: number;
  worker_memory_gb: number;
  gpus_per_worker: number;
  hint: string;
};

export const WORKER_SIZE_PRESETS: WorkerSizePreset[] = [
  {
    id: "standard",
    label: "Standard",
    worker_cpu: 4,
    worker_memory_gb: 8,
    gpus_per_worker: 1,
    hint: "Dev / smoke tests — match a small CML profile.",
  },
  {
    id: "large",
    label: "Large GPU",
    worker_cpu: 16,
    worker_memory_gb: 64,
    gpus_per_worker: 1,
    hint: "Single-GPU nodes with ample RAM for medium models.",
  },
  {
    id: "xlarge",
    label: "XL GPU",
    worker_cpu: 32,
    worker_memory_gb: 128,
    gpus_per_worker: 1,
    hint: "High-core, high-memory GPU workers for large from-scratch runs.",
  },
  {
    id: "multi-gpu",
    label: "Multi-GPU node",
    worker_cpu: 48,
    worker_memory_gb: 256,
    gpus_per_worker: 4,
    hint: "One engine with multiple GPUs — set Ray workers to 1 unless you need many nodes.",
  },
];

export function findPreset(id: string): WorkerSizePreset | undefined {
  return WORKER_SIZE_PRESETS.find((p) => p.id === id);
}
