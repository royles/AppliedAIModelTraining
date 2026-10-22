import { ValidatedNumberField } from "../ValidatedNumberField";
import { WorkerSizePresetBar } from "./WorkerSizePresetBar";
import type { WorkerSizePreset } from "../workerSizePresets";

type Props = {
  workerPresetId: string | null;
  startBusy: boolean;
  jobWorkers: number;
  jobGpus: number;
  jobWorkerCpu: number;
  jobWorkerMemoryGb: number;
  onApplyWorkerPreset: (p: WorkerSizePreset) => void;
  onClearWorkerPreset: () => void;
  onJobWorkersChange: (n: number) => void;
  onJobGpusChange: (n: number) => void;
  onJobWorkerCpuChange: (n: number) => void;
  onJobWorkerMemoryGbChange: (n: number) => void;
};

export function JobWorkerFields({
  workerPresetId,
  startBusy,
  jobWorkers,
  jobGpus,
  jobWorkerCpu,
  jobWorkerMemoryGb,
  onApplyWorkerPreset,
  onClearWorkerPreset,
  onJobWorkersChange,
  onJobGpusChange,
  onJobWorkerCpuChange,
  onJobWorkerMemoryGbChange,
}: Props) {
  function touchSizing() {
    onClearWorkerPreset();
  }

  return (
    <div className="job-worker-fields">
      <WorkerSizePresetBar
        selectedPresetId={workerPresetId}
        disabled={startBusy}
        onApply={onApplyWorkerPreset}
      />
      <div className="architecture-form-grid architecture-form-grid-2">
        <label className="architecture-field">
          <span className="architecture-field-label">Ray / CDSW worker engines</span>
          <ValidatedNumberField value={jobWorkers} min={1} max={32} onChange={onJobWorkersChange} />
        </label>
        <label className="architecture-field">
          <span className="architecture-field-label">GPUs per engine</span>
          <ValidatedNumberField
            value={jobGpus}
            min={1}
            max={8}
            onChange={(n) => {
              touchSizing();
              onJobGpusChange(n);
            }}
          />
        </label>
        <label className="architecture-field">
          <span className="architecture-field-label">CPUs per engine</span>
          <ValidatedNumberField
            value={jobWorkerCpu}
            min={0.5}
            max={128}
            integer={false}
            onChange={(n) => {
              touchSizing();
              onJobWorkerCpuChange(n);
            }}
          />
        </label>
        <label className="architecture-field">
          <span className="architecture-field-label">Memory per engine (GiB)</span>
          <ValidatedNumberField
            value={jobWorkerMemoryGb}
            min={8}
            max={512}
            onChange={(n) => {
              touchSizing();
              onJobWorkerMemoryGbChange(n);
            }}
          />
        </label>
      </div>
      <p className="hint wizard-step-footnote">
        These values are stored in the task blueprint for this run. Set CML resource profile / accelerator IDs
        under project cluster settings below (saved on the project).
      </p>
    </div>
  );
}
