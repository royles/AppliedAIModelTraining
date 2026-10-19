"""Construct Hugging Face models/tokenizers from scratch or pretrained checkpoints."""

from __future__ import annotations

from typing import Any, Tuple

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertConfig,
    BertForSequenceClassification,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from app.training.architecture import TransformerArchitectureConfig


def _architecture_from_config(config: dict[str, Any]) -> TransformerArchitectureConfig:
    raw = config.get("architecture")
    if isinstance(raw, TransformerArchitectureConfig):
        return raw
    if isinstance(raw, dict) and raw:
        return TransformerArchitectureConfig.model_validate(raw)
    return TransformerArchitectureConfig()


def build_tokenizer(config: dict[str, Any]) -> PreTrainedTokenizerBase:
    init_mode = config.get("init_mode", "pretrained")
    arch = _architecture_from_config(config)

    tokenizer_path = config.get("tokenizer_path")
    if tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    if init_mode == "from_scratch":
        # Fallback when driver did not train BPE (pretrained_gpt2 tokenizer mode).
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if arch.vocab_size > len(tokenizer):
            tokenizer.add_tokens(
                [f"<extra_{i}>" for i in range(arch.vocab_size - len(tokenizer))]
            )
        return tokenizer

    model_name = config.get("base_model") or config.get("model_name")
    if not model_name:
        raise ValueError("base_model is required when init_mode=pretrained")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = config.get("padding_side", "right")
    return tokenizer


def build_model(
    config: dict[str, Any],
    *,
    task_type: str,
) -> PreTrainedModel:
    resume_ckpt = config.get("resume_checkpoint_path")
    if resume_ckpt:
        if task_type == "causal_lm":
            return AutoModelForCausalLM.from_pretrained(resume_ckpt)
        return AutoModelForSequenceClassification.from_pretrained(
            resume_ckpt, num_labels=config.get("num_labels", 2)
        )

    init_mode = config.get("init_mode", "pretrained")
    arch = _architecture_from_config(config)

    if init_mode == "from_scratch":
        if config.get("use_lora"):
            raise ValueError("LoRA is only supported with init_mode=pretrained.")
        if task_type == "causal_lm":
            if arch.family != "gpt2":
                raise ValueError("from_scratch causal_lm currently supports family=gpt2.")
            hf = arch.to_hf_config_dict(task_type, config.get("num_labels", 2))
            model_config = GPT2Config(**hf)
            return GPT2LMHeadModel(model_config)
        if arch.family != "bert":
            raise ValueError("from_scratch sequence_classification supports family=bert.")
        hf = arch.to_hf_config_dict(task_type, config.get("num_labels", 2))
        model_config = BertConfig(**hf)
        return BertForSequenceClassification(model_config)

    model_name = config.get("base_model") or config.get("model_name")
    if not model_name:
        raise ValueError("base_model is required when init_mode=pretrained")
    if task_type == "causal_lm":
        return AutoModelForCausalLM.from_pretrained(model_name)
    return AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=config.get("num_labels", 2)
    )


def build_model_and_tokenizer(config: dict[str, Any]) -> Tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    task_type = config.get("task_type", "causal_lm")
    tokenizer = build_tokenizer(config)
    model = build_model(config, task_type=task_type)
    if model.get_input_embeddings().num_embeddings != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model
