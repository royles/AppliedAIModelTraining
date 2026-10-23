"""Prepare tokenizer + architecture for true from-scratch training runs."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable

from app.services.log_flair import task_done_line, task_start_line
from app.services.download_progress import DownloadProgressState
from app.services.task_heartbeat import provisioning_heartbeat

from app.training.dataset_sharding import load_training_corpus_staged
from app.training.dataset_staging import shared_dataset_dirs_from_config
from app.training.lm_text_utils import resolve_prompt_format
from app.training.tokenizer_config import TokenizerBuildConfig
from app.training.tokenizer_trainer import (
    recommended_min_corpus_chars,
    train_gpt2_bpe_tokenizer,
    validate_corpus_for_bpe,
)
from app.training.transformer_config import TransformerTrainingConfig

logger = logging.getLogger(__name__)

LogFn = Callable[[str], None]


def from_scratch_resource_guidance(cfg: TransformerTrainingConfig) -> dict[str, Any]:
    tok = cfg.tokenizer
    arch = cfg.architecture
    min_chars = (
        tok.min_corpus_chars
        if tok.min_corpus_chars is not None
        else recommended_min_corpus_chars(tok.vocab_size)
    )
    params = arch.parameter_count_estimate()
    return {
        "init_mode": cfg.init_mode,
        "tokenizer_mode": tok.mode,
        "vocab_size": arch.vocab_size if tok.mode == "pretrained_gpt2" else tok.vocab_size,
        "min_corpus_characters": min_chars,
        "estimated_parameters": params,
        "recommended_max_steps": max(500, cfg.max_steps),
        "notes": [
            "Full from-scratch: random transformer weights + BPE tokenizer fit on your corpus.",
            "Use a real dataset (not built-in demo) for tokenizer training.",
            "Match architecture.vocab_size to tokenizer vocab_size after BPE training.",
            "Larger vocab and width need more data and compute (GPU workers recommended).",
        ],
    }


def prepare_from_scratch_training(
    cfg: TransformerTrainingConfig,
    train_config: dict[str, Any],
    *,
    log: LogFn | None = None,
    skip_tokenizer_if_present: bool = False,
) -> dict[str, Any]:
    if cfg.init_mode != "from_scratch":
        return train_config

    tok_cfg: TokenizerBuildConfig = cfg.tokenizer
    artifact_dir = Path(train_config["artifact_dir"])
    existing_tok = artifact_dir / "tokenizer"
    if skip_tokenizer_if_present and existing_tok.is_dir():
        train_config["tokenizer_path"] = str(existing_tok)
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(existing_tok))
        arch = train_config.get("architecture")
        arch_dict = dict(arch) if isinstance(arch, dict) else cfg.architecture.model_dump()
        arch_dict["vocab_size"] = len(tok)
        train_config["architecture"] = arch_dict
        if log:
            log(
                f"Reusing saved tokenizer from {existing_tok} (resume), "
                f"vocab_size={len(tok)}."
            )
        return train_config

    def _log(msg: str) -> None:
        logger.info(msg)
        if log:
            log(msg)

    t_corpus = time.monotonic()
    _log(task_start_line("train", "Corpus load (from-scratch)"))
    dl_progress = DownloadProgressState()
    with provisioning_heartbeat(log, "Corpus load", download_progress=dl_progress):
        texts, _ = load_training_corpus_staged(
            artifact_dir=str(artifact_dir),
            dataset_cfg=cfg.dataset.model_dump(),
            task_type=cfg.task_type,
            shared_dirs=shared_dataset_dirs_from_config(train_config),
            download_progress=dl_progress,
        )
    _log(
        task_done_line(
            "Corpus load",
            seconds=time.monotonic() - t_corpus,
            detail=f"{len(texts)} samples",
        )
    )
    setting = train_config.get("prompt_format") or cfg.prompt_format or "auto"
    train_config["prompt_format"] = resolve_prompt_format(setting, texts)
    _log(
        f"Training prompt format: {train_config['prompt_format']} "
        "(chat API uses matching inference template)."
    )

    arch = train_config.get("architecture")
    if isinstance(arch, dict):
        arch_dict = dict(arch)
    else:
        arch_dict = cfg.architecture.model_dump()

    if tok_cfg.mode == "train_bpe_from_dataset":
        if cfg.dataset.source == "builtin":
            raise ValueError(
                "Built-in demo data is too small for training a tokenizer from scratch. "
                "Choose inline text, a file under data/datasets/, or a Hugging Face dataset."
            )
        total_chars = validate_corpus_for_bpe(
            texts, tok_cfg.vocab_size, tok_cfg.min_corpus_chars
        )
        t_bpe = time.monotonic()
        _log(
            task_start_line(
                "train",
                f"BPE tokenizer ({len(texts)} ex, {total_chars:,} chars, vocab={tok_cfg.vocab_size})",
            )
        )
        with provisioning_heartbeat(log, "BPE tokenizer"):
            tokenizer = train_gpt2_bpe_tokenizer(
                texts,
                vocab_size=tok_cfg.vocab_size,
                save_dir=artifact_dir / "tokenizer",
                min_frequency=tok_cfg.min_frequency,
            )
        _log(
            task_done_line(
                "BPE tokenizer",
                seconds=time.monotonic() - t_bpe,
                detail=f"vocab={len(tokenizer)}",
            )
        )
        actual_vocab = len(tokenizer)
        arch_dict["vocab_size"] = actual_vocab
        arch_dict["bos_token_id"] = tokenizer.bos_token_id
        arch_dict["eos_token_id"] = tokenizer.eos_token_id
        arch_dict["pad_token_id"] = tokenizer.pad_token_id
        train_config["tokenizer_path"] = str(artifact_dir / "tokenizer")
        train_config["architecture"] = arch_dict
        _log(f"Tokenizer ready: vocab_size={actual_vocab}, saved under artifact_dir/tokenizer.")
    elif tok_cfg.mode == "saved_artifact":
        path = train_config.get("tokenizer_path")
        if not path or not Path(path).is_dir():
            raise ValueError("tokenizer_path must point to a saved tokenizer for saved_artifact mode.")
    else:
        # pretrained_gpt2 — legacy scratch path; architecture vocab should match GPT-2 (50257)
        train_config.pop("tokenizer_path", None)
        _log("Using pretrained GPT-2 BPE vocabulary (weights still randomly initialized).")

    return train_config
