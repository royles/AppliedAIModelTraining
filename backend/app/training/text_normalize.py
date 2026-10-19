"""Normalize training text loaded from files."""

from __future__ import annotations

import re

# Trailing tags like " (on a laptop)" added by older starter_corpus versions.
_TRAILING_PAREN_TAG = re.compile(r"\s*\([^)]{1,120}\)\s*$")


def strip_trailing_parenthetical_tags(text: str) -> str:
    """Remove a single trailing `(…)` suffix from each line (User:/Assistant: rows)."""
    out_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("User:", "Assistant:")):
            prefix, rest = stripped.split(":", 1) if ":" in stripped else ("", stripped)
            body = _TRAILING_PAREN_TAG.sub("", rest).strip()
            out_lines.append(f"{prefix}: {body}" if prefix else body)
        else:
            out_lines.append(_TRAILING_PAREN_TAG.sub("", stripped).strip())
    return "\n".join(out_lines)


def normalize_training_text(text: str) -> str:
    return strip_trailing_parenthetical_tags(text)
