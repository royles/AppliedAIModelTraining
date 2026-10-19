"""Core transformer architecture specification for models built from scratch."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

ModelFamily = Literal["gpt2", "bert"]
InitMode = Literal["from_scratch", "pretrained"]
HiddenAct = Literal["gelu", "relu", "silu", "gelu_new", "quick_gelu"]


class TransformerArchitectureConfig(BaseModel):
    """
    Defines the structural artifacts of a transformer (dimensions, depth, attention).

    These map to Hugging Face `GPT2Config` / `BertConfig` when training from scratch.
    """

    family: ModelFamily = Field(
        default="gpt2",
        description="gpt2: decoder-only causal LM (OpenAI-style chat). bert: encoder classifier.",
    )
    vocab_size: int = Field(
        default=50257,
        ge=256,
        le=256_000,
        description=(
            "Embedding matrix rows; must be >= tokenizer length. "
            "BPE on small corpora often yields vocab < target (e.g. 884 vs 2048)."
        ),
    )
    hidden_size: int = Field(
        default=384,
        ge=64,
        le=8192,
        description="Model width (d_model / n_embd).",
    )
    num_hidden_layers: int = Field(default=6, ge=1, le=128, description="Transformer block count.")
    num_attention_heads: int = Field(default=6, ge=1, le=128)
    num_key_value_heads: int | None = Field(
        default=None,
        ge=1,
        le=128,
        description="Grouped-query attention heads; defaults to num_attention_heads (MHA).",
    )
    head_dim: int | None = Field(
        default=None,
        ge=8,
        le=256,
        description="Optional explicit head dimension; otherwise hidden_size // num_attention_heads.",
    )
    intermediate_size: int | None = Field(
        default=None,
        ge=64,
        le=32768,
        description="Feed-forward inner dimension; typical default is 4 * hidden_size.",
    )
    max_position_embeddings: int = Field(
        default=1024,
        ge=128,
        le=131_072,
        description="Maximum sequence length the positional encodings support.",
    )
    hidden_act: HiddenAct = Field(default="gelu_new")
    layer_norm_eps: float = Field(default=1e-5, gt=0)
    attention_probs_dropout_prob: float = Field(default=0.1, ge=0.0, le=0.9)
    hidden_dropout_prob: float = Field(default=0.1, ge=0.0, le=0.9)
    initializer_range: float = Field(
        default=0.02,
        gt=0,
        description="Stddev for weight initialization (GPT-2 / BERT default 0.02).",
    )
    tie_word_embeddings: bool = Field(
        default=True,
        description="Share input/output embeddings for causal LMs when True.",
    )
    use_bias: bool = Field(
        default=True,
        description="Attention and MLP bias terms (GPT-2 uses Conv1D biases).",
    )
    position_embedding_type: Literal["absolute", "relative_key", "relative_key_query"] = (
        "absolute"
    )
    rotary_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Reserved for RoPE-style stacks; GPT-2/BERT builders ignore non-zero today.",
    )
    rope_theta: float = Field(default=10_000.0, gt=0)

    # Token ids for generation/serving (scratch models use GPT-2 tokenizer defaults unless overridden)
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    pad_token_id: int | None = None

    @field_validator("num_key_value_heads")
    @classmethod
    def kv_heads_valid(cls, v: int | None, info) -> int | None:
        if v is None:
            return v
        n_heads = info.data.get("num_attention_heads")
        if n_heads and v > n_heads:
            raise ValueError("num_key_value_heads cannot exceed num_attention_heads.")
        return v

    @model_validator(mode="after")
    def validate_geometry(self) -> TransformerArchitectureConfig:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads.")
        if self.head_dim is not None:
            if self.head_dim * self.num_attention_heads != self.hidden_size:
                raise ValueError("head_dim * num_attention_heads must equal hidden_size.")
        if self.num_key_value_heads is None:
            object.__setattr__(self, "num_key_value_heads", self.num_attention_heads)
        if self.intermediate_size is None:
            object.__setattr__(self, "intermediate_size", self.hidden_size * 4)
        return self

    def parameter_count_estimate(self) -> int:
        """Rough parameter estimate for planning (embeddings + blocks)."""
        v, h, l, ffn = self.vocab_size, self.hidden_size, self.num_hidden_layers, self.intermediate_size or h * 4
        embed = v * h
        block = l * (4 * h * h + 2 * h * ffn)
        return embed + block

    def to_hf_config_dict(self, task_type: str, num_labels: int) -> dict[str, Any]:
        if self.family == "gpt2":
            return {
                "vocab_size": self.vocab_size,
                "n_positions": self.max_position_embeddings,
                "n_embd": self.hidden_size,
                "n_layer": self.num_hidden_layers,
                "n_head": self.num_attention_heads,
                "n_inner": self.intermediate_size,
                "activation_function": self.hidden_act if self.hidden_act != "quick_gelu" else "gelu",
                "resid_pdrop": self.hidden_dropout_prob,
                "embd_pdrop": self.hidden_dropout_prob,
                "attn_pdrop": self.attention_probs_dropout_prob,
                "layer_norm_epsilon": self.layer_norm_eps,
                "initializer_range": self.initializer_range,
                "tie_word_embeddings": self.tie_word_embeddings,
                "bos_token_id": self.bos_token_id,
                "eos_token_id": self.eos_token_id,
            }
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
            "max_position_embeddings": self.max_position_embeddings,
            "hidden_act": self.hidden_act,
            "layer_norm_eps": self.layer_norm_eps,
            "attention_probs_dropout_prob": self.attention_probs_dropout_prob,
            "hidden_dropout_prob": self.hidden_dropout_prob,
            "initializer_range": self.initializer_range,
            "position_embedding_type": self.position_embedding_type,
            "type_vocab_size": 2,
            "num_labels": num_labels,
        }
