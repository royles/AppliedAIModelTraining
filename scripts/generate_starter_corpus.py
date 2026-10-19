#!/usr/bin/env python3
"""Write data/datasets/starter_corpus.txt with verbose User/Assistant turns for causal LM training."""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "data" / "datasets" / "starter_corpus.txt"

# Verbose Q&A about the platform and transformer training (no parenthetical tags).
SEED_TURNS: list[tuple[str, str]] = [
    (
        "What is a causal language model, and how is it different from a classifier?",
        "A causal language model is trained to predict the next token in a sequence while only "
        "looking at tokens that came before it. That left-to-right constraint is what makes the "
        "same architecture useful for chat: you append the user's message, then sample or search "
        "for a continuation that reads like an assistant reply. A sequence classifier, by contrast, "
        "reads the whole input at once and emits a single label from a fixed set of categories. "
        "On this platform, causal language modeling is the default when you want OpenAI-style "
        "completions; classification uses an encoder such as BERT with a classification head.",
    ),
    (
        "How do I train a small GPT-style model from scratch on Cloudera AI using this application?",
        "Open Model Tasks, choose a from-scratch preset, and point the dataset at a real text file "
        "such as starter_corpus.txt or your own JSONL export. The job first fits a byte-level BPE "
        "tokenizer on your corpus so the vocabulary matches your data, then randomly initializes a "
        "compact GPT-2 stack whose width and depth you set under Architecture. Ray Train runs the "
        "optimization loop while CDSW worker engines can attach as distributed GPU workers. When "
        "the job succeeds, weights and tokenizer land under data/models/job_<id> and the model "
        "registry records metadata plus a model card for review.",
    ),
    (
        "Why does the platform train a custom BPE tokenizer instead of always using GPT-2's vocabulary?",
        "GPT-2's 50k token vocabulary was built on a large web crawl; it is a poor match for "
        "domain-specific or small corpora where many merges never appear. Training BPE on your "
        "own text lets rare words and punctuation split in a way that reflects your sources, and "
        "the embedding table size can follow the actual merge count instead of a huge fixed table. "
        "The trade-off is that you need enough characters to fill the vocabulary meaningfully; "
        "otherwise training stops below your target vocab size. After BPE training, the architecture "
        "vocab_size field is updated to match the tokenizer you just created.",
    ),
    (
        "What role does Ray play during a training job in this stack?",
        "Ray provides a distributed runtime: a head process in your session exposes the global "
        "control service on port 6379, and worker processes join that cluster to offer CPU or GPU "
        "resources. Ray Train wraps your PyTorch training loop, places one copy per worker, and "
        "handles process group setup so gradients can be synchronized when you scale beyond a "
        "single device. The web UI and API stay on the driver session; they stream framework logs "
        "while workers provision, attach, and run the train loop. If Ray cannot reach the head on "
        "localhost, check that start.py launched ray start --head successfully.",
    ),
    (
        "How do CDSW or CML worker engines connect to the Ray head?",
        "When you request GPU workers for a job, the platform calls the Cloudera workers API to "
        "start isolated engines in your project. Each engine runs bootstrap code that installs "
        "dependencies if needed, then executes ray start --block --address=<master_ip>:6379 so the "
        "process stays attached until the job ends. Workers discover the master IP from environment "
        "variables such as CDSW_MASTER_IP while the session driver talks to the same head on "
        "127.0.0.1 to avoid pod networking issues. Framework logs list each worker id and "
        "scheduling state so you can tell whether engines are still pending or already joined.",
    ),
    (
        "What hyperparameters should I start with for a tiny from-scratch GPT-2 smoke run?",
        "Use a few hundred to a few thousand max_steps, a modest learning rate around 6e-4, and "
        "per-device batch sizes of two to four depending on GPU memory. Keep sequences short at "
        "first so iteration is fast, then increase max_seq_length when you move to serious runs. "
        "Enable gradient checkpointing if you hit out-of-memory errors on the worker. Cosine "
        "learning-rate decay with a short warmup fraction is the default because it stabilizes "
        "early optimization and still anneals smoothly near the end of training.",
    ),
    (
        "How do I evaluate a model after training and promote it responsibly?",
        "When training completes, the registry creates an entry in the trained state with job "
        "metadata and an auto-generated model card. Call the OpenAI-compatible chat endpoint with "
        "low temperature and prompts formatted as User and Assistant turns to match training. "
        "Transition the registry entry to testing while you run qualitative checks, record benchmark "
        "metrics under benchmarked if you have them, and move to production only after you accept "
        "the limitations section of the model card. Retire older versions when a new checkpoint "
        "supersedes them so /v1/models lists only supported ids.",
    ),
    (
        "What is LoRA fine-tuning and when would I use it instead of full from-scratch training?",
        "Low-Rank Adaptation inserts small trainable matrices into attention projections while "
        "keeping most of the base weights frozen. That dramatically cuts optimizer memory and "
        "often reaches good task quality with far fewer trainable parameters. It is appropriate "
        "when you already have a strong pretrained checkpoint and want to specialize it on a "
        "narrow dataset. From-scratch training on this platform builds both tokenizer and weights "
        "anew and is mainly for education, smoke tests, or domains where no suitable base model "
        "exists. LoRA requires init_mode pretrained with a Hugging Face base_model id.",
    ),
    (
        "How should I format training text so chat inference behaves predictably?",
        "Store multi-turn examples with explicit User and Assistant lines, ideally one complete "
        "exchange per paragraph with a blank line between examples. Avoid repetitive trailing "
        "parenthetical tags that the model may memorize as filler. At inference time the server "
        "maps OpenAI roles into the same User and Assistant prefix pattern and stops generation "
        "when a new User turn would begin. Questions should be clear and answers should read like "
        "complete paragraphs rather than bullet fragments so the model learns fluent continuations.",
    ),
    (
        "What should I do when a training job fails or the framework restarts?",
        "Failed jobs remain in the task list with framework and training logs attached; read the "
        "last training error before restarting. After an API restart, in-flight jobs are marked "
        "failed automatically and you can restart with the same parameters, optionally resuming "
        "from the last epoch checkpoint if one was written. Common failures include Ray head not "
        "listening on port 6379, vocabulary size validation mismatches, or worker engines that "
        "never leave scheduling. Fix the root cause, pull the latest platform code, and run a "
        "shorter max_steps job to verify the pipeline before scaling workers again.",
    ),
]


def format_turn(user: str, assistant: str) -> str:
    return f"User: {user}\nAssistant: {assistant}"


def generate_blocks(target_chars: int) -> list[str]:
    blocks: list[str] = []
    total = 0
    i = 0
    while total < target_chars:
        u, a = SEED_TURNS[i % len(SEED_TURNS)]
        block = format_turn(u, a)
        blocks.append(block)
        total += len(block) + 2
        i += 1
    return blocks


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
    blocks = generate_blocks(args.min_chars)
    text = "\n\n".join(blocks) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(
        f"Wrote {len(blocks)} conversation blocks, {len(text):,} characters → {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
