"""Instrumented mlx-lm Qwen3 forward that exposes per-layer hidden states.

mlx-lm's stock Model.__call__ returns only final logits. DSpark needs the RAW
outputs of selected decoder layers (target_layer_ids) plus the pre-lm_head last
hidden — both for cache generation and for the eval verifier. This replicates
mlx-lm's Qwen3Model forward loop (all submodules are public) and captures them.

Validated bit-exact against the stock forward in the M1 hidden-capture spike.
target_layer_ids must exclude the final layer (raw pre-norm layer outputs vs the
normalized last hidden — mirrors assert_no_final_target_layer in base_evaluator.py).
"""

from __future__ import annotations

import mlx.core as mx
from mlx_lm.models.base import create_attention_mask


def model_dims(model):
    a = model.args
    return {
        "hidden_size": a.hidden_size,
        "num_hidden_layers": a.num_hidden_layers,
        "vocab_size": a.vocab_size,
        "head_dim": getattr(a, "head_dim", a.hidden_size // a.num_attention_heads),
        "num_attention_heads": a.num_attention_heads,
        "num_key_value_heads": a.num_key_value_heads,
        "tie_word_embeddings": a.tie_word_embeddings,
    }


def target_embed_and_head(model):
    """Return (embed_weight, lm_head_weight) from an mlx-lm target, tie-aware.

    Tied targets (Qwen3-0.6B/1.7B) reuse the embedding as the output head; untied
    targets (4B+) have a distinct lm_head. This is what the draft's
    initialize_from_target needs so it is correct across the whole family.
    """
    embed = model.model.embed_tokens.weight
    if model.args.tie_word_embeddings:
        return embed, embed
    return embed, model.lm_head.weight


def capture_hidden_states(model, input_ids: mx.array, target_layer_ids):
    """Run a cache-free forward and capture DSpark's target features.

    Args:
        model: an mlx-lm Qwen3 Model (from mlx_lm.load).
        input_ids: [B, S] int token ids.
        target_layer_ids: sorted layer indices (0-based), excluding the final layer.

    Returns dict of:
        target_hidden_states      [B, S, len(ids)*H]  bf16  (concat raw layer outputs)
        target_last_hidden_states [B, S, H]           bf16  (post-final-norm hidden)
        logits                    [B, S, V]
    """
    a = model.args
    Ln = a.num_hidden_layers
    target_layer_ids = [int(x) for x in target_layer_ids]
    assert target_layer_ids == sorted(target_layer_ids), "target_layer_ids must be sorted"
    assert all(x == -1 or 0 <= x < Ln for x in target_layer_ids), "layer ids in {-1} U [0, Ln)"
    assert max(target_layer_ids) < Ln - 1, "target_layer_ids must exclude the final layer"

    m = model.model
    h = m.embed_tokens(input_ids)
    captured = {}
    if -1 in target_layer_ids:            # -1 = embedding output (matches extract_context_feature)
        captured[-1] = h
    mask = create_attention_mask(h, None)
    for i, layer in enumerate(m.layers):
        h = layer(h, mask, None)
        if i in target_layer_ids:
            captured[i] = h
    last_hidden = m.norm(h)
    if a.tie_word_embeddings:
        logits = m.embed_tokens.as_linear(last_hidden)
    else:
        logits = model.lm_head(last_hidden)

    target_hidden_states = mx.concatenate([captured[i] for i in target_layer_ids], axis=-1)
    return {
        "target_hidden_states": target_hidden_states.astype(mx.bfloat16),
        "target_last_hidden_states": last_hidden.astype(mx.bfloat16),
        "logits": logits,
    }
