import pytest

from app.training.transformer_config import TransformerTrainingConfig


def test_from_scratch_causal_lm_requires_gpt2_family():
    with pytest.raises(ValueError, match="from_scratch causal_lm"):
        TransformerTrainingConfig(
            init_mode="from_scratch",
            task_type="causal_lm",
            architecture={"family": "bert"},
        )


def test_from_scratch_classifier_requires_bert_family():
    with pytest.raises(ValueError, match="from_scratch sequence_classification"):
        TransformerTrainingConfig(
            init_mode="from_scratch",
            task_type="sequence_classification",
            architecture={"family": "gpt2"},
        )


def test_prompt_format_default_auto():
    cfg = TransformerTrainingConfig()
    assert cfg.prompt_format == "auto"
