"""Dataset preview caps Hugging Face loads for UI responsiveness."""

from app.training.dataset_config import TrainingDatasetConfig
from app.api.routes import _preview_dataset_config


def test_hf_preview_caps_max_samples():
    cfg = TrainingDatasetConfig(
        source="huggingface",
        hf_dataset="Salesforce/wikitext",
        hf_config="wikitext-2-raw-v1",
    )
    patched, capped, limit = _preview_dataset_config(cfg)
    assert capped is True
    assert limit == 400
    assert patched.max_samples == 400


def test_hf_preview_respects_small_user_max():
    cfg = TrainingDatasetConfig(
        source="huggingface",
        hf_dataset="Salesforce/wikitext",
        hf_config="wikitext-2-raw-v1",
        max_samples=50,
    )
    patched, capped, limit = _preview_dataset_config(cfg)
    assert patched.max_samples == 50
    assert limit == 50
