import type { PromptFormatSetting } from "../api";

type TaskType = "causal_lm" | "sequence_classification";

type Props = {
  taskType: TaskType;
  onTaskTypeChange: (t: TaskType) => void;
  promptFormat: PromptFormatSetting;
  onPromptFormatChange: (p: PromptFormatSetting) => void;
  section?: "both" | "task" | "prompt";
};

const TASK_LABELS: Record<TaskType, string> = {
  causal_lm: "Causal LM (chat / completion)",
  sequence_classification: "Sequence classification",
};

const PROMPT_LABELS: Record<PromptFormatSetting, string> = {
  auto: "Auto (infer from dataset)",
  raw_lm: "Plain text continuation (raw_lm)",
  platform_user_assistant: "User / Assistant turns (platform_user_assistant)",
};

export function ModelTaskTrainingOptions({
  taskType,
  onTaskTypeChange,
  promptFormat,
  onPromptFormatChange,
  section = "both",
}: Props) {
  const showTask = section === "both" || section === "task";
  const showPrompt = section === "both" || section === "prompt";

  return (
    <div className="model-task-training-options">
      {showTask && (
        <>
          <label>
            Task type
            <select
              value={taskType}
              onChange={(e) => onTaskTypeChange(e.target.value as TaskType)}
            >
              {(Object.keys(TASK_LABELS) as TaskType[]).map((key) => (
                <option key={key} value={key}>
                  {TASK_LABELS[key]}
                </option>
              ))}
            </select>
          </label>
          <p className="hint">
            Causal LM uses GPT-2–style decoders and supports registry chat. Classification uses BERT-style
            encoders and is not served via chat.
          </p>
        </>
      )}
      {showPrompt && taskType === "causal_lm" && (
        <>
          <label>
            Inference prompt format
            <select
              value={promptFormat}
              onChange={(e) => onPromptFormatChange(e.target.value as PromptFormatSetting)}
            >
              {(Object.keys(PROMPT_LABELS) as PromptFormatSetting[]).map((key) => (
                <option key={key} value={key}>
                  {PROMPT_LABELS[key]}
                </option>
              ))}
            </select>
          </label>
          <p className="hint">
            Controls how OpenAI-style chat messages map to model input. Use <code>raw_lm</code> for
            WikiText-style corpora; use User/Assistant for demo chat corpora. Auto picks from a sample of
            your dataset at job start.
          </p>
        </>
      )}
    </div>
  );
}
