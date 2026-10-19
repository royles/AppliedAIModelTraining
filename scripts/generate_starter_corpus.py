#!/usr/bin/env python3
"""Write data/datasets/starter_corpus.txt with varied causal-LM training lines."""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "data" / "datasets" / "starter_corpus.txt"

# Representative User/Assistant turns for from-scratch BPE + tiny GPT-2 smoke runs.
SEED_TURNS: list[tuple[str, str]] = [
    (
        "What is a causal language model?",
        "It predicts the next token given all previous tokens, which is how chat assistants generate replies.",
    ),
    (
        "How do I train from scratch on this platform?",
        "Pick a text dataset, train a BPE tokenizer on that corpus, then initialize a small GPT-2 stack and run Ray Train.",
    ),
    (
        "Why use byte-level BPE?",
        "It builds subword units from raw bytes, handles punctuation and rare words, and scales to a target vocabulary size.",
    ),
    (
        "What does Ray Train do here?",
        "It launches distributed training workers, coordinates checkpoints, and streams metrics back to the driver.",
    ),
    (
        "How do CDSW workers fit in?",
        "They provision extra engines on Cloudera AI that join the Ray cluster as GPU or CPU workers for a job.",
    ),
    (
        "What is LoRA fine-tuning?",
        "Low-rank adapters train a thin set of matrices on top of a frozen base model, saving memory versus full updates.",
    ),
    (
        "Explain attention heads.",
        "Each head learns a different relationship pattern; hidden size must divide evenly by the head count.",
    ),
    (
        "What is the feed-forward block?",
        "After attention, a two-layer MLP expands hidden size to an inner dimension (often 4×) and projects back.",
    ),
    (
        "How should I pick vocab size?",
        "Smaller vocabs train faster on tiny corpora; larger vocabs need more unique text to avoid empty merges.",
    ),
    (
        "What learning rate works for scratch GPT-2?",
        "Higher rates like 6e-4 are common for small random-init models; fine-tunes usually use 2e-5 to 5e-5.",
    ),
    (
        "What is max_steps?",
        "It caps optimizer steps for smoke tests; production runs use thousands to millions depending on data.",
    ),
    (
        "Can I use Hugging Face datasets?",
        "Yes—set source to Hugging Face, provide the dataset id, split, and text column in Model Tasks.",
    ),
    (
        "What is starter_corpus.txt?",
        "A bundled line-delimited chat corpus under data/datasets for local and CAI smoke training.",
    ),
    (
        "How do I expose a model on /v1/chat/completions?",
        "After training succeeds, register an OpenAI-compatible model id and call the platform inference routes.",
    ),
    (
        "What is gradient checkpointing?",
        "It recomputes activations during the backward pass to trade extra compute for lower GPU memory.",
    ),
    (
        "Why tie word embeddings?",
        "Sharing input and output embeddings reduces parameters for causal LMs when vocab matches the LM head.",
    ),
    (
        "What is warmup in the LR schedule?",
        "A short ramp from zero stabilizes early steps before cosine or linear decay.",
    ),
    (
        "How do I debug a failed Ray job?",
        "Open the task detail view: framework logs show Ray status; training logs show step loss and errors.",
    ),
    (
        "What batch size should I use on CPU?",
        "Start with per-device batch 2 and short sequences; increase when memory allows.",
    ),
    (
        "What is sequence classification mode?",
        "An encoder model like BERT attaches a classifier head for labels instead of next-token prediction.",
    ),
]

TOPIC_VARIATIONS = [
    "on a laptop",
    "in a CDSW session",
    "with one GPU worker",
    "for a demo project",
    "before scaling out",
    "with gradient accumulation",
    "using the tiny GPT-2 preset",
    "after confirming the recipe steps",
    "with a JSONL file",
    "from inline text",
]


def format_turn(user: str, assistant: str, suffix: str) -> str:
    return f"User: {user} {suffix}\nAssistant: {assistant} {suffix}"


def generate_lines(target_chars: int) -> list[str]:
    lines: list[str] = []
    total = 0
    i = 0
    while total < target_chars:
        u, a = SEED_TURNS[i % len(SEED_TURNS)]
        suffix = TOPIC_VARIATIONS[(i // len(SEED_TURNS)) % len(TOPIC_VARIATIONS)]
        line = format_turn(u, a, f"({suffix})")
        lines.append(line)
        total += len(line) + 1
        i += 1
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min-chars",
        type=int,
        default=52_000,
        help="Minimum total characters (default covers vocab_size=2048 BPE heuristic).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT_PATH,
        help="Output path (default: data/datasets/starter_corpus.txt).",
    )
    args = parser.parse_args()
    lines = generate_lines(args.min_chars)
    text = "\n".join(lines) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"Wrote {len(lines)} lines, {len(text):,} characters → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
