"""M6 deliverable: measure DSpark acceptance_length natively in MLX.

Builds the draft, measures acceptance_length for a RANDOM-init draft (baseline ~1.0),
overfits it on the canary cache, then re-measures (should climb well above 1.0). This
is the native-MLX version of the paper's Table-1 accept-length column.

Run:
    python deepspec_mlx/scripts/eval_mlx.py --steps 40 --max-new-tokens 32
"""

from __future__ import annotations

import argparse
import os
import sys

import mlx.core as mx
import numpy as np

sys.path.insert(0, __file__.rsplit("/deepspec_mlx/", 1)[0])
from deepspec_mlx.data import CacheReader
from deepspec_mlx.modeling import build_draft_config, Qwen3DSparkModel, target_embed_and_head
from deepspec_mlx.eval import TargetRunner, generate
from deepspec_mlx.trainer import overfit


def eval_metrics(model, draft, tli, prompts, *, block_size, max_new_tokens, temperature,
                 confidence_threshold=0.0):
    """Oracle-style aggregation across prompts (matches base_evaluator.py:469-490)."""
    accept_sum = proposal_len_sum = proposal_count = 0
    pos_accept = [0] * block_size
    pos_total = [0] * block_size
    for p in prompts:
        target = TargetRunner(model, tli)                 # fresh cache per prompt
        r = generate(target, draft, p, max_new_tokens=max_new_tokens, block_size=block_size,
                     temperature=temperature, stop_ids=[], confidence_threshold=confidence_threshold, seed=0)
        accept_sum += r.accept_sum
        proposal_len_sum += r.proposal_len_sum
        proposal_count += r.proposal_count
        for k in range(block_size):
            pos_accept[k] += r.pos_accept[k]
            pos_total[k] += r.pos_total[k]
    pc = max(proposal_count, 1)
    return {
        "acceptance_length": accept_sum / pc,                          # accept_sum / proposal_count
        "verify_rate": accept_sum / max(proposal_len_sum + proposal_count, 1),
        "draft_tokens_per_proposal": proposal_len_sum / pc,
        "accept_rate_at_k": [pos_accept[k] / pos_total[k] if pos_total[k] else 0.0
                             for k in range(block_size)],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="~/dspark_mlx/cache/qwen3_0_6b_canary")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--num-draft-layers", type=int, default=3)
    ap.add_argument("--num-anchors", type=int, default=64)
    ap.add_argument("--block-size", type=int, default=7)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--n-prompts", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    args = ap.parse_args()

    from mlx_lm import load
    model, tok = load(args.model)
    r = CacheReader(os.path.expanduser(args.cache))
    tli = [int(x) for x in r.target_layer_ids]
    samples = [r[i] for i in range(len(r))]
    # prompts: first ~half of each cached sequence
    prompts = []
    for s in samples[: args.n_prompts]:
        ids = s["input_ids"]
        cut = max(4, ids.shape[0] // 2)
        prompts.append(ids[None, :cut])

    cfg = build_draft_config(model.args, dict(
        num_draft_layers=args.num_draft_layers, target_layer_ids=tli, block_size=args.block_size,
        mask_token_id=151669, num_anchors=args.num_anchors, markov_rank=256,
        markov_head_type="vanilla", confidence_head_alpha=1.0, confidence_head_with_markov=True))
    compute_dtype = mx.bfloat16 if args.dtype == "bf16" else mx.float32
    draft = Qwen3DSparkModel(cfg, compute_dtype=compute_dtype)
    draft.initialize_from_target(*target_embed_and_head(model))
    mx.eval(draft.parameters())

    def run(tag):
        m = eval_metrics(model, draft, tli, prompts, block_size=args.block_size,
                         max_new_tokens=args.max_new_tokens, temperature=args.temperature)
        akk = ",".join(f"{x:.2f}" for x in m["accept_rate_at_k"])
        print(f"  {tag}: acceptance_length={m['acceptance_length']:.3f}  "
              f"verify_rate={m['verify_rate']:.3f}  "
              f"draft_tokens/proposal={m['draft_tokens_per_proposal']:.2f}")
        print(f"       accept_rate@k = [{akk}]")
        return m

    print(f"eval: {len(prompts)} prompts, block_size={args.block_size}, "
          f"max_new_tokens={args.max_new_tokens}, temp={args.temperature}, dtype={args.dtype}")
    base = run("RANDOM-init draft")

    print(f"training {len(samples)} samples for {args.steps} steps...")
    overfit(draft, samples, steps=args.steps, lr=args.lr, log_every=max(args.steps // 2, 1))

    trained = run("TRAINED draft    ")
    print(f"  (block_size={args.block_size}, so max acceptance_length ~= {args.block_size + 1})")

    ok = trained["acceptance_length"] > 1.3 and trained["acceptance_length"] > base["acceptance_length"]
    print(f"\nRESULT: {'PASS — DSpark accept_len > 1 natively in MLX' if ok else 'FAIL'}")
    r.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
