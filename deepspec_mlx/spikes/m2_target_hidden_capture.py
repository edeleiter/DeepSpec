"""M1 spike #2 — target per-layer hidden-state capture.

mlx-lm's stock `Model.__call__` returns only final logits. Both DSpark cache-gen
(prepare_target_cache.py) and the eval verifier need the RAW outputs of selected
decoder layers (`target_layer_ids`) plus the pre-`lm_head` last hidden — the torch
reference gets these via `register_forward_hook` on `backbone.layers[layer_id]`.

This spike replicates mlx-lm's Qwen3 forward loop by hand (all submodules are
public: `model.model.{embed_tokens,layers,norm}`, `model.lm_head`), captures:
  - target_hidden_states      = concat of raw h after each target_layer_id  [1,S, n*H]
  - target_last_hidden_states = post-final-norm hidden                      [1,S, H]
  - logits                    = lm_head(last_hidden)                        [1,S, V]
and PROVES fidelity by asserting the captured logits equal the stock model()'s.

Note on `assert_no_final_target_layer` (base_evaluator.py:100): we capture raw
pre-norm layer outputs for every target layer, and the post-norm hidden separately
as target_last_hidden_states — so target_layer_ids must exclude the final layer
(its raw output would otherwise be confused with the normalized last hidden). This
spike picks target_layer_ids strictly below the final layer.

Run:
    python deepspec_mlx/spikes/m2_target_hidden_capture.py
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--prompt", default="The capital of France is Paris.")
    ap.add_argument("--layers", default="", help="comma ints; default = 5 evenly spaced, excl. final")
    args = ap.parse_args()

    import mlx.core as mx
    import numpy as np
    from mlx_lm import load
    from mlx_lm.models.base import create_attention_mask

    model, tok = load(args.model)
    a = model.args
    H, Ln, V = a.hidden_size, a.num_hidden_layers, a.vocab_size
    head_dim = getattr(a, "head_dim", H // a.num_attention_heads)
    print(f"== {args.model} ==")
    print(f"  hidden={H} layers={Ln} vocab={V} head_dim={head_dim} "
          f"heads={a.num_attention_heads} kv_heads={a.num_key_value_heads} "
          f"tie_word_embeddings={a.tie_word_embeddings}")

    if args.layers:
        target_layer_ids = [int(x) for x in args.layers.split(",")]
    else:
        # 5 evenly spaced layers in [1, Ln-2] — excludes the final layer (Ln-1).
        target_layer_ids = sorted({max(1, round(i * (Ln - 2) / 4)) for i in range(5)})
    assert max(target_layer_ids) < Ln - 1, "target_layer_ids must exclude the final layer"
    print(f"  target_layer_ids={target_layer_ids}")

    ids = mx.array(tok.encode(args.prompt))[None]
    S = ids.shape[1]

    # --- captured forward: replicate Qwen3Model.__call__ (cache=None path) ---
    m = model.model
    h = m.embed_tokens(ids)
    mask = create_attention_mask(h, None)
    captured = {}
    for i, layer in enumerate(m.layers):
        h = layer(h, mask, None)
        if i in target_layer_ids:
            captured[i] = h
    last_hidden = m.norm(h)                       # target_last_hidden_states
    if a.tie_word_embeddings:
        logits_cap = m.embed_tokens.as_linear(last_hidden)
    else:
        logits_cap = model.lm_head(last_hidden)

    target_hidden_states = mx.concatenate([captured[i] for i in target_layer_ids], axis=-1)
    mx.eval(target_hidden_states, last_hidden, logits_cap)

    # --- fidelity: captured logits must equal the stock forward ---
    logits_ref = model(ids)
    mx.eval(logits_ref)
    d = float(mx.max(mx.abs(logits_cap.astype(mx.float32) - logits_ref.astype(mx.float32))))

    print("\n== captured shapes ==")
    print(f"  target_hidden_states     {tuple(target_hidden_states.shape)}  "
          f"(expect [1,{S},{len(target_layer_ids)*H}])")
    print(f"  target_last_hidden_states {tuple(last_hidden.shape)}  (expect [1,{S},{H}])")
    print(f"  logits                    {tuple(logits_cap.shape)}  (expect [1,{S},{V}])")

    print("\n== fidelity vs stock model() ==")
    print(f"  max|Δ| captured-logits vs stock: {d:.6g}  (must be ~0)")

    ok = (
        tuple(target_hidden_states.shape) == (1, S, len(target_layer_ids) * H)
        and tuple(last_hidden.shape) == (1, S, H)
        and d < 1e-3
    )
    print(f"\nRESULT: {'PASS — per-layer hidden capture faithful' if ok else 'FAIL — investigate above'}")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ImportError as e:
        print(f"[import error] {e}", file=sys.stderr)
        sys.exit(2)
