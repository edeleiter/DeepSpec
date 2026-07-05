"""Instrumented Qwen3.5 (Ornith) TEXT backbone that exposes per-layer hidden states.

Ornith-1.0-9B is a VLM (Qwen3_5ForConditionalGeneration); mlx_lm.load returns a Model
whose `.language_model` is the text `TextModel`. Its backbone `Qwen3_5TextModel` is HYBRID:
~3/4 layers are linear GatedDeltaNet (is_linear=True), 1/4 full attention, and it applies
TWO masks — `ssm_mask` for linear layers, `fa_mask` for full (qwen3_5.py:254-275).

This replicates that forward with `cache=None` (cache-free — the only correct path for the
linear layers, whose recurrent state can't be rewound) and captures raw per-layer hidden at
`target_layer_ids`. Analogue of qwen3_target_capture.py. Validated bit-exact against the
stock `model.language_model(input_ids)` forward.
"""

from __future__ import annotations

import mlx.core as mx
from mlx_lm.models.base import create_attention_mask, create_ssm_mask


def ornith_text_model(model):
    """The text TextModel (has .model backbone, .args, .lm_head) from an mlx-lm Ornith Model."""
    return getattr(model, "language_model", model)


def model_dims(model):
    lm = ornith_text_model(model)
    a = lm.args
    return {
        "hidden_size": a.hidden_size,
        "num_hidden_layers": a.num_hidden_layers,
        "vocab_size": a.vocab_size,
        "head_dim": a.head_dim,
        "num_attention_heads": a.num_attention_heads,
        "num_key_value_heads": a.num_key_value_heads,
        "tie_word_embeddings": a.tie_word_embeddings,
        "full_attention_interval": a.full_attention_interval,
    }


def _capture(model, tli, input_ids):
    lm = ornith_text_model(model)
    a = lm.args
    m = lm.model                                   # Qwen3_5TextModel
    h = m.embed_tokens(input_ids)
    captured = {}
    if -1 in tli:                                  # -1 = embedding output
        captured[-1] = h
    fa_mask = create_attention_mask(h, None)
    ssm_mask = create_ssm_mask(h, None)
    want = set(tli)
    for i, layer in enumerate(m.layers):
        mask = ssm_mask if layer.is_linear else fa_mask
        h = layer(h, mask=mask, cache=None)
        if i in want:
            captured[i] = h
    last = m.norm(h)
    logits = m.embed_tokens.as_linear(last) if a.tie_word_embeddings else lm.lm_head(last)
    target_hidden = mx.concatenate([captured[i] for i in tli], axis=-1)
    return logits, last, target_hidden


def capture_forward(model, tli, input_ids):
    """Runner hook: (model, tli, input_ids) -> (logits [1,S,V], target_hidden [1,S,L*H])."""
    tli = [int(x) for x in tli]
    logits, _, target_hidden = _capture(model, tli, input_ids)
    return logits, target_hidden


def capture_hidden_states(model, input_ids, target_layer_ids):
    """Cache-gen entry: dict of target_hidden_states / target_last_hidden_states / logits."""
    a = ornith_text_model(model).args
    Ln = a.num_hidden_layers
    tli = [int(x) for x in target_layer_ids]
    assert tli == sorted(tli), "target_layer_ids must be sorted"
    assert all(x == -1 or 0 <= x < Ln for x in tli), "layer ids in {-1} U [0, Ln)"
    assert max(tli) < Ln - 1, "target_layer_ids must exclude the final layer"
    logits, last, target_hidden = _capture(model, tli, input_ids)
    return {
        "target_hidden_states": target_hidden.astype(mx.bfloat16),
        "target_last_hidden_states": last.astype(mx.bfloat16),
        "logits": logits,
    }
