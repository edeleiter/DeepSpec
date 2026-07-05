"""M5 overfit test: train the DSpark draft on the tiny canary cache and watch the
accept_rate climb off its untrained ~0.005 floor. This proves the training loop
converges end-to-end (forward + loss + Muon/AdamW + schedule + freezing).

Run:
    python deepspec_mlx/scripts/overfit_canary.py \
        --cache ~/dspark_mlx/cache/qwen3_0_6b_canary --steps 60
"""

from __future__ import annotations

import argparse
import sys

import mlx.core as mx

sys.path.insert(0, __file__.rsplit("/deepspec_mlx/", 1)[0])
from deepspec_mlx.data import CacheReader
from deepspec_mlx.modeling import build_draft_config, Qwen3DSparkModel, target_embed_and_head
from deepspec_mlx.trainer import overfit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="~/dspark_mlx/cache/qwen3_0_6b_canary")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--num-draft-layers", type=int, default=3)
    ap.add_argument("--num-anchors", type=int, default=64)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    args = ap.parse_args()

    from mlx_lm import load
    import os

    model, tok = load(args.model)
    r = CacheReader(os.path.expanduser(args.cache))
    tli = [int(x) for x in r.target_layer_ids]
    samples = [r[i] for i in range(len(r))]
    print(f"cache: {len(samples)} samples, target_layer_ids={tli}, hidden={r.hidden_size}")

    cfg = build_draft_config(model.args, dict(
        num_draft_layers=args.num_draft_layers, target_layer_ids=tli, block_size=7,
        mask_token_id=151669, num_anchors=args.num_anchors, markov_rank=256,
        markov_head_type="vanilla", confidence_head_alpha=1.0,
        confidence_head_with_markov=True))
    compute_dtype = mx.bfloat16 if args.dtype == "bf16" else mx.float32
    draft = Qwen3DSparkModel(cfg, compute_dtype=compute_dtype)
    draft.initialize_from_target(*target_embed_and_head(model))
    mx.eval(draft.parameters())

    print(f"overfitting {len(samples)} samples for {args.steps} steps (lr={args.lr})...")
    history = overfit(draft, samples, steps=args.steps, lr=args.lr, log_every=10)

    ce0, ar0 = history[0][1], history[0][2]
    ceN, arN = history[-1][1], history[-1][2]
    print(f"\nsummary: ce {ce0:.3f} -> {ceN:.3f}   accept_rate {ar0:.4f} -> {arN:.4f}")
    ok = ceN < ce0 - 1.0 and arN > ar0 * 3
    print(f"RESULT: {'PASS — draft is learning (ce down, accept_rate up)' if ok else 'FAIL — no clear learning'}")
    r.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
