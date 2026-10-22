"""Chat prompts aligned with platform training data (User:/Assistant:)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from transformers import PreTrainedTokenizerBase

# Matches starter_corpus.txt and typical from-scratch causal_lm jobs.
PLATFORM_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' or message['role'] == 'user' %}"
    "User: {{ message['content'] }}\n"
    "{% elif message['role'] == 'assistant' %}"
    "Assistant: {{ message['content'] }}\n"
    "{% endif %}"
    "{% endfor %}"
    "Assistant: "
)


def platform_messages_to_prompt(messages: list[dict[str, str]]) -> str:
    """OpenAI roles → User:/Assistant: lines used during training."""
    lines: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        if role in ("system", "user"):
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
    lines.append("Assistant:")
    return "\n".join(lines)


def read_platform_config(model_path: str | Path) -> dict[str, Any]:
    path = Path(model_path) / "platform_config.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def use_platform_prompt_format(model_path: str | Path, meta: dict[str, Any]) -> bool:
    return meta.get("prompt_format") == "platform_user_assistant"


def raw_lm_messages_to_prompt(messages: list[dict[str, str]]) -> str:
    """
    Plain continuation prompt for corpus-trained LMs (WikiText, etc.).

    Uses the latest user turn (and optional system prefix) without User:/Assistant:
    tokens the model was not trained on.
    """
    system_parts: list[str] = []
    user_parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            system_parts.append(content)
        elif role == "user":
            user_parts.append(content)
    prefix = "\n\n".join(system_parts).strip()
    user = user_parts[-1] if user_parts else ""
    if prefix and user:
        return f"{prefix}\n\n{user}"
    return user or prefix


def messages_to_prompt(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, str]],
    *,
    model_path: str | Path,
) -> str:
    meta = read_platform_config(model_path)
    if use_platform_prompt_format(model_path, meta):
        if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
            try:
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:  # noqa: BLE001
                pass
        return platform_messages_to_prompt(messages)

    if meta.get("prompt_format") == "raw_lm" or (
        meta.get("init_mode") == "from_scratch" and meta.get("task_type") == "causal_lm"
    ):
        return raw_lm_messages_to_prompt(messages)

    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    parts: list[str] = []
    for msg in messages:
        parts.append(f"{msg['role']}: {msg['content']}")
    parts.append("assistant:")
    return "\n".join(parts)


def ensure_platform_chat_template(tokenizer: PreTrainedTokenizerBase) -> None:
    if not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = PLATFORM_CHAT_TEMPLATE
