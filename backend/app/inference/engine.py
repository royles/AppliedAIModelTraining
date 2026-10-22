"""Load trained transformers and serve OpenAI-style chat completions."""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from app.inference.generation import platform_generation_extras, raw_lm_generation_extras
from app.inference.prompting import messages_to_prompt, read_platform_config, use_platform_prompt_format
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


def chat_complete(
    model_path: str,
    messages: list[dict[str, str]],
    *,
    temperature: float = 1.0,
    max_tokens: int = 256,
    top_p: float = 1.0,
) -> tuple[str, UsageInfo]:
    tokenizer, model = load_causal_model(model_path)
    prompt = messages_to_prompt(tokenizer, messages, model_path=model_path)
    inputs = tokenizer(prompt, return_tensors="pt").to(_device())
    prompt_len = inputs["input_ids"].shape[1]

    gen_kwargs = _build_generation_kwargs(
        tokenizer, model_path, temperature=temperature, max_tokens=max_tokens, top_p=top_p
    )

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


def _build_generation_kwargs(
    tokenizer: AutoTokenizer,
    model_path: str,
    *,
    temperature: float,
    max_tokens: int,
    top_p: float,
) -> dict:
    meta = read_platform_config(model_path)
    platform_model = use_platform_prompt_format(model_path, meta)
    raw_lm = meta.get("prompt_format") == "raw_lm" or (
        meta.get("init_mode") == "from_scratch"
        and meta.get("task_type") == "causal_lm"
        and not platform_model
    )
    gen_kwargs: dict = {
        "max_new_tokens": max_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id or tokenizer.pad_token_id,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = max(temperature, 1e-5)
        gen_kwargs["top_p"] = top_p
    else:
        gen_kwargs["do_sample"] = False
    if platform_model:
        gen_kwargs.update(platform_generation_extras(tokenizer))
    elif raw_lm:
        gen_kwargs.update(raw_lm_generation_extras(tokenizer))
    return gen_kwargs


def chat_complete_stream(
    model_path: str,
    messages: list[dict[str, str]],
    *,
    temperature: float = 1.0,
    max_tokens: int = 256,
    top_p: float = 1.0,
) -> Iterator[str]:
    """Yield decoded completion text chunks (OpenAI-style streaming)."""
    tokenizer, model = load_causal_model(model_path)
    prompt = messages_to_prompt(tokenizer, messages, model_path=model_path)
    inputs = tokenizer(prompt, return_tensors="pt").to(_device())
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = _build_generation_kwargs(
        tokenizer, model_path, temperature=temperature, max_tokens=max_tokens, top_p=top_p
    )
    generation_kwargs = {**inputs, **gen_kwargs, "streamer": streamer}
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    try:
        for piece in streamer:
            if piece:
                yield piece
    finally:
        thread.join(timeout=600)
