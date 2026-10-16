import type { TrainingDatasetConfig, TokenizerBuildConfig } from "./api";

/** Sane defaults for a local / CAI from-scratch smoke run that can pass the training recipe. */
export const DEFAULT_PRESET = "scratch_gpt2_tiny";

export const DEFAULT_DATASET: TrainingDatasetConfig = {
  source: "text_file",
  path: "starter_corpus.txt",
  repeat: 1,
};

export const DEFAULT_TOKENIZER: TokenizerBuildConfig = {
  mode: "train_bpe_from_dataset",
  vocab_size: 2048,
  min_frequency: 2,
};

export const DEFAULT_VOCAB_SIZE = 2048;
export const DEFAULT_HIDDEN_SIZE = 256;
export const DEFAULT_LAYERS = 4;
export const DEFAULT_HEADS = 4;
export const DEFAULT_MAX_STEPS = 120;
export const DEFAULT_PROJECT_NAME = "My models";
export const DEFAULT_JOB_NAME = "From-scratch run";
export const DEFAULT_OPENAI_MODEL_ID = "my-model-chat";
