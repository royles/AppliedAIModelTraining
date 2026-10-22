import { useEffect, useState } from "react";
import { api, type TokenizerBuildConfig } from "../api";

const DEBOUNCE_MS = 500;

type Args = {
  initMode: "from_scratch" | "pretrained";
  tokenizer: TokenizerBuildConfig;
  vocabSize: number;
  hiddenSize: number;
  layers: number;
  heads: number;
  maxSteps: number;
};

/** Min corpus for BPE — does not load the dataset on the server. */
export function useDebouncedMinCorpusChars(args: Args) {
  const { initMode, tokenizer, vocabSize, hiddenSize, layers, heads, maxSteps } = args;
  const [minCorpusChars, setMinCorpusChars] = useState<number | null>(null);

  useEffect(() => {
    if (initMode !== "from_scratch") {
      setMinCorpusChars(null);
      return;
    }

    let cancelled = false;
    const timer = window.setTimeout(() => {
      api
        .fromScratchGuidance({
          init_mode: "from_scratch",
          task_type: "causal_lm",
          architecture: {
            family: "gpt2",
            vocab_size: vocabSize,
            hidden_size: hiddenSize,
            num_hidden_layers: layers,
            num_attention_heads: heads,
            max_position_embeddings: 512,
            tie_word_embeddings: true,
          },
          tokenizer,
          max_steps: maxSteps,
          dataset: { source: "builtin", builtin_name: "platform_demo" },
        })
        .then((g) => {
          if (!cancelled) {
            setMinCorpusChars((g.min_corpus_characters as number) ?? null);
          }
        })
        .catch(() => {
          if (!cancelled) setMinCorpusChars(null);
        });
    }, DEBOUNCE_MS);

    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [initMode, tokenizer, vocabSize, hiddenSize, layers, heads, maxSteps]);

  return minCorpusChars;
}
