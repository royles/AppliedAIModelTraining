import type { TokenizerBuildConfig } from "./api";
import { recommendedMinCorpusChars } from "./localFromScratchHints";
import { ValidatedNumberField } from "./ValidatedNumberField";

type Props = {
  section?: "mode" | "tokenizer" | "all";
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
  section = "all",
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
  function patchTokenizer(partial: Partial<TokenizerBuildConfig>) {
    const next = { ...tokenizer, ...partial };
    onTokenizerChange(next);
    if (partial.vocab_size !== undefined) {
      onVocabSizeChange(partial.vocab_size);
    }
  }

  const showMode = section === "all" || section === "mode";
  const showTokenizer = section === "all" || section === "tokenizer";

  const wrapper = section === "all" ? "article" : "div";
  const Wrapper = wrapper as "article" | "div";

  const tokVocab = tokenizer.vocab_size ?? vocabSize;
  const minCorpus = recommendedMinCorpusChars(tokVocab);

  return (
    <Wrapper className={section === "all" ? "card" : "blueprint-section-fields"}>
      {showMode && (
        <>
          {section === "all" && <h3>Training mode</h3>}
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
          {initMode === "pretrained" && (
            <p className="hint">
              Fine-tuning loads Hugging Face weights from the preset&apos;s <code>base_model</code>.
              Use a pretrained preset (e.g. causal_lm_small) or pass <code>base_model</code> via API.
            </p>
          )}
        </>
      )}

      {showTokenizer && initMode === "from_scratch" && (
        <>
          <p className="hint">
            True from-scratch builds train a <strong>BPE tokenizer on your dataset</strong> when the
            job runs. Corpus size is checked at run time (rough guide: ≥{minCorpus.toLocaleString()}{" "}
            characters for this vocab).
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
                <ValidatedNumberField
                  value={tokenizer.vocab_size ?? vocabSize}
                  min={256}
                  max={256_000}
                  onChange={(n) => {
                    patchTokenizer({ vocab_size: n });
                    onVocabSizeChange(n);
                  }}
                />
              </label>
              <label>
                BPE min frequency
                <ValidatedNumberField
                  value={tokenizer.min_frequency ?? 2}
                  min={1}
                  max={10_000}
                  onChange={(n) => patchTokenizer({ min_frequency: n })}
                />
              </label>
            </div>
          )}
          <div className="guidance-box">
            <p className="hint">
              Geometry snapshot: d={hiddenSize}, L={layers}, max_steps={maxSteps.toLocaleString()}
              (parameter estimate updates on the Architecture step when available).
            </p>
          </div>
        </>
      )}

      {showTokenizer && initMode === "pretrained" && (
        <p className="hint">Tokenizer comes from the pretrained checkpoint preset.</p>
      )}
    </Wrapper>
  );
}
