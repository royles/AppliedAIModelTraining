"""Generation helpers for small platform-trained causal LMs."""

from __future__ import annotations

import torch
from transformers import PreTrainedTokenizerBase, StoppingCriteria, StoppingCriteriaList


class StopOnTokenSequences(StoppingCriteria):
    """Stop when the generated suffix matches a stop string (e.g. new User: turn)."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, stop_strings: tuple[str, ...]) -> None:
        self.stop_ids: list[list[int]] = []
        for s in stop_strings:
            ids = tokenizer.encode(s, add_special_tokens=False)
            if ids:
                self.stop_ids.append(ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:  # noqa: ARG002
        seq = input_ids[0].tolist()
        for stop in self.stop_ids:
            n = len(stop)
            if n and len(seq) >= n and seq[-n:] == stop:
                return True
        return False


def platform_generation_extras(
    tokenizer: PreTrainedTokenizerBase,
) -> dict:
    """Defaults that reduce parenthesis / phrase loops on tiny from-scratch models."""
    stops = StopOnTokenSequences(tokenizer, ("\nUser:", "\nAssistant:", "<|endoftext|>"))
    return {
        "repetition_penalty": 1.12,
        "stopping_criteria": StoppingCriteriaList([stops]),
    }
