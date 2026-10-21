import { WORKER_SIZE_PRESETS, type WorkerSizePreset } from "../workerSizePresets";

type Props = {
  selectedPresetId: string | null;
  onApply: (preset: WorkerSizePreset) => void;
  disabled?: boolean;
};

export function WorkerSizePresetBar({ selectedPresetId, onApply, disabled }: Props) {
  return (
    <div className="worker-size-presets" role="group" aria-label="Worker size presets">
      <p className="hint worker-size-presets-intro">
        Pick a sizing preset that matches a <strong>CML resource profile</strong> on your cluster. CPU
        and memory must not exceed the profile limits; every engine gets the GPU count you set.
      </p>
      <div className="worker-size-preset-buttons">
        {WORKER_SIZE_PRESETS.map((p) => (
          <button
            key={p.id}
            type="button"
            className={`btn-secondary btn-compact ${selectedPresetId === p.id ? "worker-size-preset-active" : ""}`}
            disabled={disabled}
            title={p.hint}
            onClick={() => onApply(p)}
          >
            {p.label}
          </button>
        ))}
      </div>
    </div>
  );
}
