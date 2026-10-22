"""Causal LM text normalization, chunking, and prompt-format detection."""

from __future__ import annotations

import re
from typing import Literal

PromptFormat = Literal["platform_user_assistant", "raw_lm"]
PromptFormatSetting = Literal["auto", "platform_user_assistant", "raw_lm"]

_MIN_CHUNK_CHARS = 32
_DEFAULT_CHUNK_CHARS = 1800


def normalize_causal_lm_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text_for_lm(text: str, *, max_chars: int) -> list[str]:
    """Split long documents into paragraph-bounded chunks for LM training."""
    text = normalize_causal_lm_text(text)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    buf = ""
    for para in paragraphs:
        if len(para) > max_chars:
            if buf:
                chunks.append(buf)
                buf = ""
            for start in range(0, len(para), max_chars):
                piece = para[start : start + max_chars].strip()
                if len(piece) >= _MIN_CHUNK_CHARS:
                    chunks.append(piece)
            continue
        candidate = f"{buf}\n\n{para}".strip() if buf else para
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            if buf:
                chunks.append(buf)
            buf = para
    if buf and len(buf) >= _MIN_CHUNK_CHARS:
        chunks.append(buf)
    return chunks


def prepare_causal_lm_texts(
    texts: list[str],
    *,
    source: str,
    chunk_chars: int | None = None,
    min_chunk_chars: int = _MIN_CHUNK_CHARS,
) -> list[str]:
    """
    Normalize and optionally chunk training strings for causal LM.

    Hugging Face rows (e.g. full WikiText articles) are chunked so the model
    does not only see article titles and newline runs in the first 512 tokens.
    """
    if not texts:
        return texts

    should_chunk = source in ("huggingface", "text_file", "parquet_s3", "iceberg")
    max_chars = chunk_chars if chunk_chars is not None else _DEFAULT_CHUNK_CHARS

    out: list[str] = []
    for raw in texts:
        normalized = normalize_causal_lm_text(raw)
        if not normalized:
            continue
        if should_chunk and len(normalized) > max_chars:
            out.extend(chunk_text_for_lm(normalized, max_chars=max_chars))
        elif len(normalized) >= min_chunk_chars:
            out.append(normalized)
        elif not should_chunk and normalized:
            out.append(normalized)

    if not out:
        raise ValueError("Dataset produced zero usable causal LM examples after normalization")
    return out


def detect_prompt_format(texts: list[str]) -> PromptFormat:
    """Infer whether training data uses platform User:/Assistant: turns."""
    sample = texts[: min(200, len(texts))]
    hits = 0
    for t in sample:
        if "User:" in t and "Assistant:" in t:
            hits += 1
    if hits >= max(1, len(sample) // 8):
        return "platform_user_assistant"
    return "raw_lm"


def resolve_prompt_format(
    setting: str | None,
    texts: list[str] | None,
    *,
    fallback: PromptFormat = "raw_lm",
) -> PromptFormat:
    """Apply explicit prompt_format or infer from corpus when setting is auto."""
    if setting and setting not in ("auto", ""):
        if setting in ("raw_lm", "platform_user_assistant"):
            return setting  # type: ignore[return-value]
        raise ValueError(f"Unsupported prompt_format: {setting}")
    if texts:
        return detect_prompt_format(texts)
    return fallback
