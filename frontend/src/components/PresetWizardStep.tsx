import type { TransformerPreset } from "../api";
import { ModelTaskTrainingOptions } from "./ModelTaskTrainingOptions";
import type { PromptFormatSetting } from "../api";

type Props = {
  presets: TransformerPreset[];
  preset: string;
  onPresetChange: (name: string) => void;
  selectedPreset: TransformerPreset | undefined;
  taskType: "causal_lm" | "sequence_classification";
  onTaskTypeChange: (t: "causal_lm" | "sequence_classification") => void;
  promptFormat: PromptFormatSetting;
  onPromptFormatChange: (p: PromptFormatSetting) => void;
};

export function PresetWizardStep(props: Props) {
  const { presets, preset, onPresetChange, selectedPreset, ...taskProps } = props;
  return (
    <div className="wizard-step-compact">
      <p className="hint architecture-section-lead">
        Choose a scale template, then set the training objective. Use <strong>causal LM</strong> for
        GPT-style next-token pretraining or fine-tuning.
      </p>
      <label className="architecture-field">
        <span className="architecture-field-label">Model preset</span>
        <select value={preset} onChange={(e) => onPresetChange(e.target.value)}>
          {presets.map((p) => (
            <option key={p.name} value={p.name}>
              {p.title}
            </option>
          ))}
        </select>
      </label>
      {selectedPreset?.description && (
        <p className="hint architecture-field-hint">{selectedPreset.description}</p>
      )}
      <ModelTaskTrainingOptions section="both" {...taskProps} />
    </div>
  );
}
