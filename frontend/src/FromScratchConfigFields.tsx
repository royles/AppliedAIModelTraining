import { useEffect, useState } from "react";
import { api, type TokenizerBuildConfig } from "./api";

type Props = {
  initMode: "from_scratch" | "pretrained";
  onInitModeChange: (mode: "from_scratch" | "pretrained") => void;
  tokenizer: TokenizerBuildConfig;
  onTokenizerChange: (t: TokenizerBuildConfig) => void;
  vocabSize: number;
  onVocabSizeChange: (n: number) => void;
  maxSteps: number;
  hiddenSize: number;
  layers: number;
};

export function FromScratchConfigFields({
  initMode,
  onInitModeChange,
  tokenizer,
  onTokenizerChange,
  vocabSize,
  onVocabSizeChange,
  maxSteps,
  hiddenSize,
  layers,
}: Props) {
  const [guidance, setGuidance] = useState<Record<string, unknown> | null>(null);

  useEffect(() => {
    if (initMode !== "from_scratch") {
      setGuidance(null);
      return;
    }
    api
      .fromScratchGuidance({
        init_mode: "from_scratch",
        task_type: "causal_lm",
        architecture: {
          family: "gpt2",
          vocab_size: vocabSize,
          hidden_size: hiddenSize,
          num_hidden_layers: layers,
          num_attention_heads: 4,
          max_position_embeddings: 512,
          tie_word_embeddings: true,
        },
        tokenizer,
        max_steps: maxSteps,
        max_seq_length: 256,
        per_device_train_batch_size: 2,
        learning_rate: 6e-4,
      })
      .then(setGuidance)
      .catch(() => setGuidance(null));
  }, [initMode, tokenizer, vocabSize, maxSteps, hiddenSize, layers]);

  function patchTokenizer(partial: Partial<TokenizerBuildConfig>) {
    const next = { ...tokenizer, ...partial };
    onTokenizerChange(next);
    if (partial.vocab_size !== undefined) {
      onVocabSizeChange(partial.vocab_size);
    }
  }

  return (
    <article className="card">
      <h3>Training mode</h3>
      <label>
        Initialization
        <select
          value={initMode}
          onChange={(e) =>
            onInitModeChange(e.target.value as "from_scratch" | "pretrained")
          }
        >
          <option value="from_scratch">Full from scratch (random weights)</option>
          <option value="pretrained">Fine-tune pretrained checkpoint</option>
        </select>
      </label>

      {initMode === "from_scratch" && (
        <>
          <p className="hint">
            True from-scratch builds train a <strong>BPE tokenizer on your dataset</strong> and
            randomly initialize the transformer to match that vocabulary.
          </p>
          <label>
            Tokenizer
            <select
              value={tokenizer.mode}
              onChange={(e) =>
                patchTokenizer({
                  mode: e.target.value as TokenizerBuildConfig["mode"],
                })
              }
            >
              <option value="train_bpe_from_dataset">
                Train BPE on dataset (recommended)
              </option>
              <option value="pretrained_gpt2">
                Use GPT-2 vocabulary only (legacy)
              </option>
            </select>
          </label>
          {tokenizer.mode === "train_bpe_from_dataset" && (
            <div className="arch-grid arch-grid-compact">
              <label>
                Target vocab size
                <input
                  type="number"
                  min={256}
                  max={256000}
                  value={tokenizer.vocab_size ?? vocabSize}
                  onChange={(e) =>
                    patchTokenizer({ vocab_size: Number(e.target.value) })
                  }
                />
              </label>
              <label>
                BPE min frequency
                <input
                  type="number"
                  min={1}
                  value={tokenizer.min_frequency ?? 2}
                  onChange={(e) =>
                    patchTokenizer({ min_frequency: Number(e.target.value) })
                  }
                />
              </label>
            </div>
          )}
          {guidance && (
            <div className="guidance-box">
              <p className="hint">
                Min corpus size:{" "}
                <strong>
                  {(guidance.min_corpus_characters as number)?.toLocaleString()} characters
                </strong>
                · Est. parameters:{" "}
                <strong>
                  {(guidance.estimated_parameters as number)?.toLocaleString()}
                </strong>
              </p>
              <ul className="guidance-notes">
                {(guidance.notes as string[])?.map((n) => (
                  <li key={n}>{n}</li>
                ))}
              </ul>
            </div>
          )}
        </>
      )}

      {initMode === "pretrained" && (
        <p className="hint">
          Fine-tuning loads Hugging Face weights from the preset&apos;s <code>base_model</code>.
          Use a pretrained preset (e.g. causal_lm_small) or pass <code>base_model</code> via API.
        </p>
      )}
    </article>
  );
}
