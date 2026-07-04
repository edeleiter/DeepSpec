"""DSpark draft config builder for Qwen3.5-family targets (e.g. Ornith-1.0-9B).

The DSpark draft is a small, plain full-attention transformer (see
``deepspec/modeling/dspark/qwen3/modeling.py``); it only *consumes* the target's
cached hidden states as attention context. So we do NOT port Qwen3.5's hybrid
linear/SSM attention, output gating, or multiway RoPE into the draft. We only
need a clean *text* config whose fields the Qwen3 draft modules understand.

Two things make Qwen3.5 different from the Qwen3 builder (it is closer to Gemma4):

  * Layout: Qwen3.5 is a multimodal composite whose text fields are nested under
    ``target_config.text_config`` (verified via scripts/ornith/check_load.py;
    at the top level num_hidden_layers/vocab_size/head_dim/layer_types/rope are
    all absent). So we deepcopy ``text_config``, exactly like the Gemma4 builder.
  * RoPE: Qwen3.5 uses partial rotary (``partial_rotary_factor``) and multiway
    interleaved RoPE. The draft builds a plain ``Qwen3RotaryEmbedding`` and
    applies full ``rotate_half``, which requires cos/sin sized to the full
    ``head_dim``. We therefore sanitize the RoPE fields to standard full-dim
    RoPE. The draft is trained from scratch, so this is correct, not lossy.

Run ``scripts/ornith/check_load.py`` first to confirm the real field values this
builder assumes (rope layout, hidden_act, attention_bias, layer indices).
"""

import copy

from deepspec.modeling.dspark.common import validate_target_layer_ids


# Ornith uses head_dim=256, at which flex_attention's triton kernel exceeds
# shared memory on consumer Blackwell GPUs (RTX 50xx). Use eager attention (plain
# matmul+softmax, no shared-memory limit); the draft's attention tensors are tiny
# so the cost is negligible. See create_dspark_attention_bias for the dense mask.
TRAIN_ATTN_IMPLEMENTATION = "eager"

# Field names on the Qwen3.5 config that encode partial / multiway RoPE. They
# must be neutralized so the draft's plain full-dim RoPE is well defined.
_MROPE_FIELDS = ("mrope_section", "mrope_interleaved", "rope_sections")


def _resolve_rope_theta(text_config) -> float:
    rope_parameters = getattr(text_config, "rope_parameters", None)
    if isinstance(rope_parameters, dict) and "rope_theta" in rope_parameters:
        return float(rope_parameters["rope_theta"])
    return float(getattr(text_config, "rope_theta", 1.0e7))


def get_qwen35_text_config(target_config):
    """Deepcopy the nested Qwen3.5 text config (the draft is text-only)."""
    model_type = str(getattr(target_config, "model_type", ""))
    assert model_type == "qwen3_5", (
        "Qwen3.5 DSpark expects a top-level qwen3_5 target config, "
        f"got model_type={model_type!r}."
    )
    text_config = getattr(target_config, "text_config", None)
    assert text_config is not None, (
        "Qwen3.5 DSpark expects target_config.text_config (the multimodal "
        "composite's nested text model config)."
    )
    text_config = copy.deepcopy(text_config)
    # Defensive: if a vision block ever rides along on the text config, drop it
    # so nothing downstream tries to build a vision tower from the draft config.
    if getattr(text_config, "vision_config", None) is not None:
        try:
            delattr(text_config, "vision_config")
        except AttributeError:
            text_config.vision_config = None
    return text_config


def _sanitize_rope(text_config) -> None:
    """Force plain, full-head_dim RoPE for the draft (kills partial/multiway RoPE)."""
    theta = _resolve_rope_theta(text_config)
    # A plain default rope config: rope_type 'default' + theta, with NO
    # partial_rotary_factor or mrope keys, so Qwen3RotaryEmbedding builds cos/sin
    # over the full head_dim and apply_rotary_pos_emb's full rotate_half matches.
    clean = {"rope_type": "default", "rope_theta": theta}
    text_config.partial_rotary_factor = 1.0
    text_config.rope_theta = theta
    # In this transformers, rope_parameters is linked to rope_scaling: setting
    # rope_scaling=None also nulls rope_parameters, which makes the draft's
    # Qwen3RotaryEmbedding crash on rope_parameters["rope_type"]. So set BOTH to
    # the clean dict rather than nulling either.
    for field in ("rope_scaling", "rope_parameters"):
        if hasattr(text_config, field):
            try:
                setattr(text_config, field, dict(clean))
            except (AttributeError, TypeError):
                pass
    for field in _MROPE_FIELDS:
        if hasattr(text_config, field):
            setattr(text_config, field, None)


def _ensure_qwen3_fields(text_config) -> None:
    """Backfill fields the Qwen3 draft modules read but Qwen3.5 may omit."""
    defaults = {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "hidden_act": "silu",
        "initializer_range": 0.02,
        "sliding_window": None,
    }
    for name, value in defaults.items():
        if getattr(text_config, name, None) is None:
            setattr(text_config, name, value)


def build_draft_config(target_config, model_args):
    draft_config = get_qwen35_text_config(target_config)
    _sanitize_rope(draft_config)
    _ensure_qwen3_fields(draft_config)

    num_target_layers = int(draft_config.num_hidden_layers)
    num_draft_layers = int(model_args.num_draft_layers)
    layer_types = ["full_attention"] * num_draft_layers

    assert "target_layer_ids" in model_args, "target_layer_ids must be provided."
    target_layer_ids = validate_target_layer_ids(
        model_args.target_layer_ids,
        num_target_layers,
    )

    confidence_head_alpha = float(model_args.confidence_head_alpha)
    assert confidence_head_alpha >= 0.0
    enable_confidence_head = confidence_head_alpha > 0.0
    if enable_confidence_head:
        assert "confidence_head_with_markov" in model_args, (
            "confidence_head_with_markov must be provided when "
            "confidence_head_alpha > 0."
        )

    markov_rank = int(model_args.markov_rank)
    assert markov_rank >= 0, f"markov_rank must be >= 0, got {markov_rank}"
    if markov_rank > 0:
        assert "markov_head_type" in model_args, (
            "markov_head_type must be provided when markov_rank > 0."
        )

    draft_config.architectures = ["Qwen35DSparkModel"]
    draft_config.target_model_type = str(getattr(target_config, "model_type", "qwen3_5"))
    draft_config.num_target_layers = num_target_layers
    draft_config.num_hidden_layers = num_draft_layers
    draft_config.block_size = int(model_args.block_size)
    draft_config.tie_word_embeddings = False
    draft_config.layer_types = layer_types
    draft_config._attn_implementation = TRAIN_ATTN_IMPLEMENTATION
    draft_config.mask_token_id = int(model_args.mask_token_id)
    draft_config.target_layer_ids = target_layer_ids
    draft_config.num_anchors = int(model_args.num_anchors)
    draft_config.enable_confidence_head = enable_confidence_head
    if enable_confidence_head:
        draft_config.confidence_head_with_markov = bool(
            model_args.confidence_head_with_markov
        )
    draft_config.markov_rank = markov_rank
    if markov_rank > 0:
        draft_config.markov_head_type = str(model_args.markov_head_type)
    return draft_config


__all__ = [
    "build_draft_config",
    "get_qwen35_text_config",
]
