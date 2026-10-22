"""Hugging Face capped loads must not require full shard downloads."""

from unittest.mock import MagicMock, patch

from app.training.dataset_config import TrainingDatasetConfig
from app.training.dataset_loader import _load_huggingface


def test_hf_capped_load_uses_split_slice():
    cfg = TrainingDatasetConfig(
        source="huggingface",
        hf_dataset="Salesforce/wikitext",
        hf_config="wikitext-2-raw-v1",
        max_samples=100,
    )
    ds = [{"text": "hello world"}] * 3

    with patch("datasets.load_dataset", return_value=ds) as load_dataset:
        texts, labels = _load_huggingface(cfg)

    load_dataset.assert_called_once()
    assert load_dataset.call_args.kwargs.get("split") == "train[:100]"
    assert labels is None
    assert len(texts) == 3


def test_hf_full_load_without_max_samples():
    cfg = TrainingDatasetConfig(
        source="huggingface",
        hf_dataset="Salesforce/wikitext",
        hf_config="wikitext-2-raw-v1",
    )
    ds = MagicMock()
    ds.__len__.return_value = 2
    ds.__iter__.return_value = iter([{"text": "a"}, {"text": "b"}])

    with patch("datasets.load_dataset", return_value=ds) as load_dataset:
        texts, _ = _load_huggingface(cfg)

    assert load_dataset.call_args.kwargs.get("split") == "train"
    assert texts == ["a", "b"]
