"""Tokenizer construction for from-scratch vs fine-tune training."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

TokenizerMode = Literal["train_bpe_from_dataset", "pretrained_gpt2", "saved_artifact"]


class TokenizerBuildConfig(BaseModel):
    """
    train_bpe_from_dataset: fit byte-level BPE on the job dataset (true from-zero vocab).
    pretrained_gpt2: reuse OpenAI GPT-2 BPE (weights still random if init_mode=from_scratch).
    saved_artifact: load tokenizer from tokenizer_path (retrain / resume).
    """

    mode: TokenizerMode = Field(
        default="train_bpe_from_dataset",
        description="How to obtain tokenization for from-scratch jobs.",
    )
    vocab_size: int = Field(
        default=8192,
        ge=256,
        le=256_000,
        description="Target vocabulary size when training BPE from the corpus.",
    )
    min_frequency: int = Field(default=2, ge=1, le=100)
    min_corpus_chars: int | None = Field(
        default=None,
        ge=1000,
        description="Override minimum total characters required to train BPE.",
    )
