"""
Transformer training configuration aligned with common fine-tuning best practices.

References: Hugging Face Trainer defaults, LLM fine-tuning guides (warmup, cosine
schedule, weight decay, gradient clipping), and PEFT/LoRA for parameter-efficient
adaptation.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from app.training.architecture import InitMode, TransformerArchitectureConfig  # noqa: TC001
from app.training.dataset_config import TrainingDatasetConfig
from app.training.tokenizer_config import TokenizerBuildConfig


TaskType = Literal["causal_lm", "sequence_classification"]


class TransformerTrainingConfig(BaseModel):
    """Validated hyperparameters and architecture options for transformer jobs."""

    init_mode: InitMode = Field(
        default="from_scratch",
        description="from_scratch builds weights from TransformerArchitectureConfig; pretrained loads base_model.",
    )
    task_type: TaskType = Field(
        default="causal_lm",
        description="causal_lm for chat/OpenAI-style models; sequence_classification for classifiers.",
    )
    architecture: TransformerArchitectureConfig = Field(
        default_factory=TransformerArchitectureConfig,
        description="Core model artifacts (width, depth, heads, FFN, positions) when building from scratch.",
    )
    base_model: str | None = Field(
        default=None,
        description="Hugging Face checkpoint when init_mode=pretrained.",
    )
    openai_model_id: str | None = Field(
        default=None,
        description="Optional id exposed via GET/POST /v1/models and chat completions after training.",
    )

    # Tokenization & sequence
    max_seq_length: int = Field(default=512, ge=32, le=8192)
    padding_side: Literal["left", "right"] = Field(
        default="right",
        description="Left padding is often preferred for decoder-only generation batches.",
    )
    truncation: bool = True

    # Optimization (AdamW + schedule)
    learning_rate: float = Field(default=2e-5, gt=0, le=1.0)
    weight_decay: float = Field(default=0.01, ge=0.0, le=1.0)
    adam_beta1: float = Field(default=0.9, ge=0.0, lt=1.0)
    adam_beta2: float = Field(default=0.999, ge=0.0, lt=1.0)
    adam_epsilon: float = Field(default=1e-8, gt=0)
    max_grad_norm: float = Field(default=1.0, gt=0, description="Gradient clipping norm.")

    warmup_ratio: float = Field(
        default=0.03,
        ge=0.0,
        le=0.5,
        description="Fraction of total steps used for linear LR warmup.",
    )
    lr_scheduler_type: Literal["linear", "cosine", "constant"] = Field(
        default="cosine",
        description="Cosine decay with warmup is a common default for transformer fine-tunes.",
    )

    max_steps: int = Field(default=100, ge=1, le=5_000_000)
    per_device_train_batch_size: int = Field(default=4, ge=1, le=512)
    gradient_accumulation_steps: int = Field(
        default=1,
        ge=1,
        le=128,
        description="Effective batch = per_device_batch * accumulation * num_workers.",
    )
    logging_steps: int = Field(default=5, ge=1)

    # Regularization & stability
    label_smoothing: float = Field(default=0.0, ge=0.0, lt=1.0)
    attention_dropout: float | None = Field(
        default=None,
        ge=0.0,
        le=0.9,
        description="If set, overrides the model config attention dropout.",
    )
    hidden_dropout: float | None = Field(default=None, ge=0.0, le=0.9)

    # Memory / throughput
    gradient_checkpointing: bool = Field(
        default=False,
        description="Trades compute for memory; recommended for large models.",
    )
    fp16: bool = False
    bf16: bool = False

    # Parameter-efficient fine-tuning (LoRA)
    use_lora: bool = Field(
        default=False,
        description="Enable LoRA adapters instead of full-weight updates when supported.",
    )
    lora_r: int = Field(default=16, ge=1, le=256)
    lora_alpha: int = Field(default=32, ge=1, le=512)
    lora_dropout: float = Field(default=0.05, ge=0.0, le=0.5)
    lora_target_modules: list[str] | None = Field(
        default=None,
        description="Auto-detected per architecture when omitted (e.g. q_proj,v_proj for LLMs).",
    )

    # Classification-only
    num_labels: int = Field(default=2, ge=2, le=512)

    dataset: TrainingDatasetConfig = Field(
        default_factory=TrainingDatasetConfig,
        description="Training corpus (builtin, inline, files under data/datasets, or Hugging Face).",
    )
    tokenizer: TokenizerBuildConfig = Field(
        default_factory=TokenizerBuildConfig,
        description="Tokenizer pipeline (train BPE on dataset vs pretrained GPT-2 vocab).",
    )

    @field_validator("bf16")
    @classmethod
    def bf16_not_with_fp16(cls, v: bool, info) -> bool:
        if v and info.data.get("fp16"):
            raise ValueError("Enable only one of fp16 or bf16.")
        return v

    @model_validator(mode="after")
    def validate_init_mode(self) -> TransformerTrainingConfig:
        if self.init_mode == "pretrained" and not self.base_model:
            raise ValueError("base_model is required when init_mode=pretrained.")
        if self.init_mode == "from_scratch" and self.use_lora:
            raise ValueError("use_lora requires init_mode=pretrained.")
        if self.init_mode == "from_scratch" and self.tokenizer.mode == "train_bpe_from_dataset":
            if self.architecture.vocab_size != self.tokenizer.vocab_size:
                object.__setattr__(self.architecture, "vocab_size", self.tokenizer.vocab_size)
        if self.init_mode == "from_scratch":
            if self.task_type == "causal_lm" and self.architecture.family != "gpt2":
                raise ValueError("from_scratch causal_lm requires architecture.family=gpt2.")
            if self.task_type == "sequence_classification" and self.architecture.family != "bert":
                raise ValueError(
                    "from_scratch sequence_classification requires architecture.family=bert."
                )
        if self.attention_dropout is not None:
            self.architecture.attention_probs_dropout_prob = self.attention_dropout
        if self.hidden_dropout is not None:
            self.architecture.hidden_dropout_prob = self.hidden_dropout
        return self

    def effective_batch_size(self, num_workers: int) -> int:
        return (
            self.per_device_train_batch_size
            * self.gradient_accumulation_steps
            * max(1, num_workers)
        )

    def warmup_steps(self) -> int:
        return max(1, int(self.max_steps * self.warmup_ratio))

    def to_train_dict(self, *, job_id: int, artifact_dir: str) -> dict[str, Any]:
        base = self.base_model or f"scratch-{self.architecture.family}"
        return {
            **self.model_dump(),
            "job_id": job_id,
            "artifact_dir": artifact_dir,
            "model_name": base,
            "batch_size": self.per_device_train_batch_size,
        }

    def to_json(self) -> str:
        return json.dumps(self.model_dump())

    @classmethod
    def from_json(cls, raw: str | None) -> TransformerTrainingConfig:
        if not raw:
            return cls()
        return cls.model_validate(json.loads(raw))

    @classmethod
    def merge_preset(
        cls, preset_name: str | None, overrides: dict[str, Any] | None
    ) -> TransformerTrainingConfig:
        base = get_preset(preset_name or "scratch_gpt2_tiny").model_copy()
        if overrides:
            base = base.model_validate({**base.model_dump(), **overrides})
        return base

    @classmethod
    def from_legacy_job_fields(
        cls,
        *,
        model_name: str,
        batch_size: int,
        learning_rate: float,
        max_steps: int,
        epochs: int,
        config_json: str | None,
    ) -> TransformerTrainingConfig:
        if config_json:
            cfg = cls.from_json(config_json)
            return cfg
        task = "sequence_classification" if "bert" in model_name.lower() else "causal_lm"
        return cls(
            init_mode="pretrained",
            task_type=task,
            base_model=model_name,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            max_steps=max_steps,
        )


class TransformerPreset(BaseModel):
    name: str
    title: str
    description: str
    config: TransformerTrainingConfig


TRANSFORMER_PRESETS: dict[str, TransformerPreset] = {
    "scratch_gpt2_tiny": TransformerPreset(
        name="scratch_gpt2_tiny",
        title="From-scratch tiny GPT-2 (chat)",
        description="Randomly initialized decoder-only transformer sized for local CPU dev.",
        config=TransformerTrainingConfig(
            init_mode="from_scratch",
            task_type="causal_lm",
            tokenizer=TokenizerBuildConfig(
                mode="train_bpe_from_dataset",
                vocab_size=8192,
            ),
            architecture=TransformerArchitectureConfig(
                family="gpt2",
                vocab_size=8192,
                hidden_size=256,
                num_hidden_layers=4,
                num_attention_heads=4,
                intermediate_size=1024,
                max_position_embeddings=512,
                tie_word_embeddings=True,
            ),
            max_seq_length=256,
            learning_rate=6e-4,
            per_device_train_batch_size=2,
            max_steps=120,
            warmup_ratio=0.05,
            weight_decay=0.1,
            lr_scheduler_type="cosine",
            dataset=TrainingDatasetConfig(
                source="builtin",
                builtin_name="platform_demo",
                repeat=40,
            ),
        ),
    ),
    "scratch_gpt2_small": TransformerPreset(
        name="scratch_gpt2_small",
        title="From-scratch small GPT-2",
        description="6-layer GPT-style stack suitable for multi-hour GPU training runs.",
        config=TransformerTrainingConfig(
            init_mode="from_scratch",
            task_type="causal_lm",
            architecture=TransformerArchitectureConfig(
                family="gpt2",
                hidden_size=768,
                num_hidden_layers=6,
                num_attention_heads=12,
                max_position_embeddings=2048,
            ),
            max_seq_length=1024,
            learning_rate=3e-4,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=4,
            max_steps=10_000,
            gradient_checkpointing=True,
            warmup_ratio=0.03,
        ),
    ),
    "causal_lm_small": TransformerPreset(
        name="causal_lm_small",
        title="Small chat model (DistilGPT-2)",
        description="Fast CPU-friendly causal LM for local dev and OpenAI-compatible chat.",
        config=TransformerTrainingConfig(
            init_mode="pretrained",
            task_type="causal_lm",
            base_model="distilgpt2",
            max_seq_length=256,
            learning_rate=5e-5,
            per_device_train_batch_size=4,
            max_steps=80,
            warmup_ratio=0.06,
            lr_scheduler_type="cosine",
            gradient_checkpointing=False,
        ),
    ),
    "causal_lm_lora": TransformerPreset(
        name="causal_lm_lora",
        title="LoRA fine-tune (GPT-2)",
        description="Parameter-efficient adaptation with LoRA; lower memory than full fine-tune.",
        config=TransformerTrainingConfig(
            init_mode="pretrained",
            task_type="causal_lm",
            base_model="gpt2",
            max_seq_length=512,
            learning_rate=2e-4,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            max_steps=500,
            use_lora=True,
            lora_r=16,
            lora_alpha=32,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            weight_decay=0.01,
        ),
    ),
    "classifier_bert": TransformerPreset(
        name="classifier_bert",
        title="Sequence classifier (DistilBERT)",
        description="Encoder classifier head; not served via chat completions.",
        config=TransformerTrainingConfig(
            init_mode="pretrained",
            task_type="sequence_classification",
            base_model="distilbert-base-uncased",
            max_seq_length=128,
            learning_rate=3e-5,
            per_device_train_batch_size=16,
            max_steps=100,
            num_labels=2,
            label_smoothing=0.1,
            warmup_ratio=0.1,
            lr_scheduler_type="linear",
        ),
    ),
    "causal_lm_gpu": TransformerPreset(
        name="causal_lm_gpu",
        title="GPU causal LM (GPT-2 medium steps)",
        description="Heavier run intended for GPU workers with mixed precision.",
        config=TransformerTrainingConfig(
            init_mode="pretrained",
            task_type="causal_lm",
            base_model="gpt2",
            max_seq_length=1024,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=2,
            max_steps=2000,
            learning_rate=2e-5,
            bf16=True,
            gradient_checkpointing=True,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            max_grad_norm=1.0,
        ),
    ),
}


def get_preset(name: str) -> TransformerTrainingConfig:
    preset = TRANSFORMER_PRESETS.get(name)
    if not preset:
        raise KeyError(f"Unknown preset: {name}")
    return preset.config.model_copy()


def list_presets() -> list[TransformerPreset]:
    return list(TRANSFORMER_PRESETS.values())
