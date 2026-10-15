"""Load trained transformers and serve OpenAI-style chat completions."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.inference.schemas_openai import UsageInfo

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_loaded: dict[str, tuple[AutoTokenizer, AutoModelForCausalLM]] = {}


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_causal_model(model_path: str) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    path = str(Path(model_path).resolve())
    with _lock:
        if path in _loaded:
            return _loaded[path]
        logger.info("Loading causal LM from %s", path)
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path)
        model.to(_device())
        model.eval()
        _loaded[path] = (tokenizer, model)
        return tokenizer, model


def unload_model(model_path: str) -> None:
    path = str(Path(model_path).resolve())
    with _lock:
        _loaded.pop(path, None)


def messages_to_prompt(tokenizer: AutoTokenizer, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    parts: list[str] = []
    for msg in messages:
        parts.append(f"{msg['role']}: {msg['content']}")
    parts.append("assistant:")
    return "\n".join(parts)


def chat_complete(
    model_path: str,
    messages: list[dict[str, str]],
    *,
    temperature: float = 1.0,
    max_tokens: int = 256,
    top_p: float = 1.0,
) -> tuple[str, UsageInfo]:
    tokenizer, model = load_causal_model(model_path)
    prompt = messages_to_prompt(tokenizer, messages)
    inputs = tokenizer(prompt, return_tensors="pt").to(_device())
    prompt_len = inputs["input_ids"].shape[1]

    gen_kwargs: dict = {
        "max_new_tokens": max_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.eos_token_id or tokenizer.pad_token_id,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = max(temperature, 1e-5)
        gen_kwargs["top_p"] = top_p

    with torch.no_grad():
        output = model.generate(**inputs, **gen_kwargs)

    new_tokens = output[0, prompt_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    completion_tokens = int(new_tokens.shape[0])
    usage = UsageInfo(
        prompt_tokens=prompt_len,
        completion_tokens=completion_tokens,
        total_tokens=prompt_len + completion_tokens,
    )
    return text, usage
