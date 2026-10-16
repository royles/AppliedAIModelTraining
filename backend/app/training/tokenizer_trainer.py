"""Train GPT-2–style byte-level BPE tokenizers on the job corpus."""

from __future__ import annotations

import logging
from pathlib import Path

from transformers import GPT2TokenizerFast

logger = logging.getLogger(__name__)

SPECIAL_TOKENS = ["<|endoftext|>", "<|pad|>"]


def recommended_min_corpus_chars(vocab_size: int) -> int:
    """Heuristic: BPE needs enough bytes to fill the vocabulary meaningfully."""
    return max(10_000, vocab_size * 25)


def validate_corpus_for_bpe(texts: list[str], vocab_size: int, min_chars: int | None) -> int:
    total = sum(len(t) for t in texts)
    required = min_chars if min_chars is not None else recommended_min_corpus_chars(vocab_size)
    if total < required:
        raise ValueError(
            f"Corpus too small to train vocab_size={vocab_size} BPE tokenizer: "
            f"{total:,} characters (need at least {required:,}). "
            "Add a larger dataset (JSONL/text/Hugging Face) or lower vocab_size."
        )
    return total


def train_gpt2_bpe_tokenizer(
    texts: list[str],
    *,
    vocab_size: int,
    save_dir: Path,
    min_frequency: int = 2,
) -> GPT2TokenizerFast:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    save_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)

    fast = GPT2TokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        unk_token="<|endoftext|>",
        pad_token="<|pad|>",
    )
    fast.save_pretrained(save_dir)
    logger.info("Saved trained BPE tokenizer to %s (len=%s)", save_dir, len(fast))
    return fast
