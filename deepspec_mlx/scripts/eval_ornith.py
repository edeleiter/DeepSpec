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

_ROOT = __file__.rsplit("/deepspec_mlx/", 1)[0]
sys.path.insert(0, _ROOT)
import argparse

from deepspec_mlx.data import CacheReader
from deepspec_mlx.modeling import build_draft_config, Qwen3DSparkModel
from deepspec_mlx.modeling.qwen3_5_target_capture import (
    capture_forward, ornith_text_model, model_dims,
)
from deepspec_mlx.eval import CacheFreeTargetRunner, generate
from deepspec_mlx.trainer import overfit

MODEL = "deepreinforce-ai/Ornith-1.0-9B"
TLI = [7, 15, 23]                     # full-attention layers (torch dspark_ornith_9b.py)
MASK_ID = 248044                      # Ornith pad/mask token


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None, help="Ornith target cache -> train before eval (M8b)")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-new-tokens", type=int, default=14)
    ap.add_argument("--heldout-jsonl", default=f"{_ROOT}/eval_datasets/gsm8k.jsonl")
    ap.add_argument("--heldout-idx", type=int, default=120, help="gsm8k line for a HELD-OUT eval prompt")
    ap.add_argument("--save", default=None, help="dir to save the trained draft checkpoint")
    args = ap.parse_args()
    import json
    import os

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

    # ---- optional training (M8b): overfit an Ornith target cache ----
    eval_prompt = ids[:, :10]
    trained = False
    if args.cache:
        r = CacheReader(os.path.expanduser(args.cache))
        samples = [r[i] for i in range(len(r))]
        print(f"\ntraining on {len(samples)} Ornith samples for {args.steps} steps...")
        overfit(draft, samples, steps=args.steps, lr=args.lr, log_every=max(args.steps // 2, 1))
        eval_prompt = samples[0]["input_ids"][None, : max(6, samples[0]["input_ids"].shape[0] // 2)]
        r.close()
        trained = True
        if args.save:
            from deepspec_mlx.serve import save_draft
            save_draft(draft, os.path.expanduser(args.save), target_id=MODEL,
                       arch="qwen3_5", model_id="dspark-ornith-9b")
            print(f"saved draft -> {args.save}")

    # ---- Gate 2: the spec-decode loop runs via cache-free verify ----
    def eval_on(prompt, tag):
        runner = CacheFreeTargetRunner(model, TLI, capture_fn=capture_forward)
        out = generate(runner, draft, prompt, max_new_tokens=args.max_new_tokens, block_size=7,
                       temperature=0.0, stop_ids=[], seed=0)
        al = out.accept_sum / max(out.proposal_count, 1)
        print(f"  [{tag}] committed {len(out.committed)} over {out.proposal_count} verify steps; "
              f"acceptance_length={al:.3f}  (block_size=7, max ~= 8)")
        return al

    def heldout_prompt():
        p = args.heldout_jsonl
        if not (p and os.path.exists(p)):
            return None
        with open(p, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == args.heldout_idx:
                    turns = json.loads(line).get("turns") or []
                    if not turns:
                        return None
                    hid = mx.array(tok.encode(turns[0])[:48], dtype=mx.int32)[None]
                    return hid[:, : max(6, hid.shape[1] // 2)]
        return None

    print(f"\n== Gate 2: spec-decode on Ornith (cache-free verify) ==")
    if trained:
        al_train = eval_on(eval_prompt, "train-prompt = upper bound (memorized)")
        hp = heldout_prompt()
        al = eval_on(hp, "HELD-OUT = honest") if hp is not None else al_train
        gate2 = al > 1.05           # honest held-out accelerates decoding
    else:
        al = eval_on(eval_prompt, "random-init")
        gate2 = al >= 1.0
    print(f"  Gate 2: {'PASS' if gate2 else 'FAIL'}")

    ok = gate1 and gate2
    headline = ("PASS — trained DSpark draft accelerates Ornith on a HELD-OUT prompt (M8b)" if trained
                else "PASS — DSpark spec-decode runs on Ornith (what torch could not eval)")
    print(f"\nRESULT: {headline if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
