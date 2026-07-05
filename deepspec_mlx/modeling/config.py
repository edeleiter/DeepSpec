"""Draft config for the MLX DSpark model — mirrors deepspec/modeling/dspark/qwen3/config.py.

Derives from an mlx-lm target ModelArgs. Plain dataclass (no HF PretrainedConfig).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DSparkDraftConfig:
    # from target
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    attention_bias: bool = False
    # draft-specific
    num_hidden_layers: int = 5           # = num_draft_layers
    num_target_layers: int = 0
    target_layer_ids: List[int] = field(default_factory=list)
    block_size: int = 7
    mask_token_id: int = 0
    num_anchors: int = 128
    markov_rank: int = 256
    markov_head_type: str = "vanilla"
    enable_confidence_head: bool = True
    confidence_head_with_markov: bool = True


def build_draft_config(target_args, model_args: dict) -> DSparkDraftConfig:
    """target_args: mlx-lm ModelArgs; model_args: dict of DSpark knobs."""
    num_target_layers = int(target_args.num_hidden_layers)
    tli = [int(x) for x in model_args["target_layer_ids"]]
    assert tli == sorted(tli) and tli, "target_layer_ids must be sorted, non-empty"
    # -1 = embedding output; otherwise a real decoder layer, excluding the final one
    # (mirrors validate_target_layer_ids + assert_no_final_target_layer in the oracle).
    assert all(x == -1 or 0 <= x < num_target_layers for x in tli), "layer ids in {-1} U [0, Ln)"
    assert max(tli) < num_target_layers - 1, "target_layer_ids must exclude the final layer"

    confidence_alpha = float(model_args.get("confidence_head_alpha", 1.0))
    head_dim = getattr(target_args, "head_dim", target_args.hidden_size // target_args.num_attention_heads)
    return DSparkDraftConfig(
        hidden_size=int(target_args.hidden_size),
        num_attention_heads=int(target_args.num_attention_heads),
        num_key_value_heads=int(target_args.num_key_value_heads),
        head_dim=int(head_dim),
        intermediate_size=int(target_args.intermediate_size),
        vocab_size=int(target_args.vocab_size),
        rms_norm_eps=float(target_args.rms_norm_eps),
        rope_theta=float(target_args.rope_theta),
        max_position_embeddings=int(target_args.max_position_embeddings),
        attention_bias=bool(getattr(target_args, "attention_bias", False)),
        num_hidden_layers=int(model_args["num_draft_layers"]),
        num_target_layers=num_target_layers,
        target_layer_ids=tli,
        block_size=int(model_args["block_size"]),
        mask_token_id=int(model_args["mask_token_id"]),
        num_anchors=int(model_args["num_anchors"]),
        markov_rank=int(model_args.get("markov_rank", 256)),
        markov_head_type=str(model_args.get("markov_head_type", "vanilla")),
        enable_confidence_head=confidence_alpha > 0.0,
        confidence_head_with_markov=bool(model_args.get("confidence_head_with_markov", True)),
    )
