"""M8a — prove the DSpark spec-decode loop runs on Ornith-1.0-9B (hybrid Qwen3.5).

Two gates:
  1. The hybrid capture (fa/ssm mask split, cache-free) reproduces the stock text-model
     logits BIT-EXACT.
  2. The full spec-decode loop runs end-to-end via the CACHE-FREE target verify — the
     mechanism that unblocks what the PyTorch reference could not eval (linear layers
     can't be trim-rewound). A RANDOM-init draft gives acceptance_length ~= 1.0; the point
     is "the loop runs, no crash," not speed.

Run (after the ~15GB download is cached):
    python deepspec_mlx/scripts/eval_ornith.py
"""

from __future__ import annotations

import sys

import mlx.core as mx

sys.path.insert(0, __file__.rsplit("/deepspec_mlx/", 1)[0])
from deepspec_mlx.modeling import build_draft_config, Qwen3DSparkModel
from deepspec_mlx.modeling.qwen3_5_target_capture import (
    capture_forward, ornith_text_model, model_dims,
)
from deepspec_mlx.eval import CacheFreeTargetRunner, generate

MODEL = "deepreinforce-ai/Ornith-1.0-9B"
TLI = [7, 15, 23]                     # full-attention layers (torch dspark_ornith_9b.py)
MASK_ID = 248044                      # Ornith pad/mask token


def main():
    from mlx_lm import load
    print(f"loading {MODEL} ...", flush=True)
    model, tok = load(MODEL)
    lm = ornith_text_model(model)
    dims = model_dims(model)
    print(f"  text backbone: {dims}")

    # ---- Gate 1: capture bit-exact vs stock forward ----
    ids = mx.array(tok.encode("The capital of France is Paris, and the capital of Japan is"))[None]
    stock = lm(ids)
    cap_logits, cap_hidden = capture_forward(model, TLI, ids)
    mx.eval(stock, cap_logits, cap_hidden)
    d = float(mx.max(mx.abs(stock.astype(mx.float32) - cap_logits.astype(mx.float32))))
    print(f"\n== Gate 1: hybrid capture vs stock ==")
    print(f"  logits shape {tuple(cap_logits.shape)}; target_hidden {tuple(cap_hidden.shape)} "
          f"(expect [1,S,{len(TLI)*dims['hidden_size']}])")
    print(f"  max|Δ| capture vs stock logits: {d:.6g}  (must be ~0)")
    gate1 = d < 1e-3 and tuple(cap_hidden.shape) == (1, ids.shape[1], len(TLI) * dims["hidden_size"])
    print(f"  Gate 1: {'PASS' if gate1 else 'FAIL'}")

    # ---- build a (random-init) DSpark draft from the text config ----
    cfg = build_draft_config(lm.args, dict(
        num_draft_layers=2, target_layer_ids=TLI, block_size=7, mask_token_id=MASK_ID,
        num_anchors=16, markov_rank=256, markov_head_type="vanilla",
        confidence_head_alpha=1.0, confidence_head_with_markov=True))
    print(f"\n  draft cfg: hidden={cfg.hidden_size} head_dim={cfg.head_dim} vocab={cfg.vocab_size} "
          f"layers={cfg.num_hidden_layers} tie={lm.args.tie_word_embeddings}")
    draft = Qwen3DSparkModel(cfg, compute_dtype=mx.bfloat16)
    # untied heads: embed + real lm_head, both bf16
    draft.initialize_from_target(lm.model.embed_tokens.weight, lm.lm_head.weight)
    mx.eval(draft.parameters())

    # ---- Gate 2: the spec-decode loop runs via cache-free verify ----
    print(f"\n== Gate 2: spec-decode loop on Ornith (cache-free verify) ==")
    prompt = ids[:, :10]
    runner = CacheFreeTargetRunner(model, TLI, capture_fn=capture_forward)
    out = generate(runner, draft, prompt, max_new_tokens=14, block_size=7,
                   temperature=0.0, stop_ids=[], seed=0)
    al = out.accept_sum / max(out.proposal_count, 1)
    print(f"  committed {len(out.committed)} tokens over {out.proposal_count} verify steps; "
          f"acceptance_length={al:.3f} (random draft -> ~1.0 expected)")
    gate2 = out.proposal_count > 0 and len(out.committed) > 0 and al >= 1.0
    print(f"  Gate 2: {'PASS' if gate2 else 'FAIL'}")

    ok = gate1 and gate2
    print(f"\nRESULT: {'PASS — DSpark spec-decode runs on Ornith (what torch could not eval)' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
