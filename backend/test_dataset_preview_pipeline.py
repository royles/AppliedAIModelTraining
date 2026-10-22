"""UI preview uses relaxed LM prep so WikiText-style rows are counted."""

from app.training.dataset_config import TrainingDatasetConfig
from app.training.dataset_loader import load_training_samples


def test_preview_keeps_short_hf_lines():
    cfg = TrainingDatasetConfig(
        source="huggingface",
        hf_dataset="demo",
        max_samples=10,
    )
    short_lines = [" = Title = ", "short", "another line of text here for training"]

    class FakeDs(list):
        pass

    fake = FakeDs({"text": t} for t in short_lines)

    from unittest.mock import patch

    with patch(
        "app.training.dataset_loader._load_huggingface",
        return_value=(short_lines, None),
    ):
        texts, _ = load_training_samples(cfg, task_type="causal_lm", for_preview=True)

    assert len(texts) >= len(short_lines)
    assert sum(len(t) for t in texts) > 0
