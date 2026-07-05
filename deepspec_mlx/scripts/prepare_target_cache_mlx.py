"""Generate a (tiny) DSpark target cache natively in MLX — no torch, no SGLang.

Loads an mlx-lm Qwen3 target, runs the instrumented capture forward over a handful
of prompts, and writes the v2 on-disk cache. For the canary this is deliberately
small; scale --num / --max-length toward a real run later (watch disk).

loss_mask is all-ones for the canary (supervise every next-token position — an
off-policy, capture-only cache). Realistic assistant-only masking / on-policy answer
generation is a refinement for later milestones.

Run:
    python deepspec_mlx/scripts/prepare_target_cache_mlx.py \
        --out ~/dspark_mlx/cache/qwen3_0_6b_canary --num 8 --max-length 128
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import mlx.core as mx

sys.path.insert(0, __file__.rsplit("/deepspec_mlx/", 1)[0])
from deepspec_mlx.data import write_target_cache
from deepspec_mlx.modeling.qwen3_target_capture import capture_hidden_states, model_dims

# Fallback canary prompts (used if --jsonl is absent). Short + varied.
DEFAULT_PROMPTS = [
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "In mathematics, a prime number is a natural number greater than one.",
    "She poured the coffee, opened her laptop, and began to write.",
    "Photosynthesis converts sunlight, water, and carbon dioxide into glucose.",
    "The capital of France is Paris; the capital of Japan is Tokyo.",
    "To sort a list efficiently, many languages use an adaptive merge sort.",
    "He walked to the store to buy milk, eggs, bread, and a newspaper.",
    "The theory of relativity reshaped our understanding of space and time.",
]


def load_prompts(jsonl, num):
    if jsonl and os.path.exists(jsonl):
        out = []
        with open(jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                turns = obj.get("turns") or []
                if turns:
                    out.append(turns[0])
                if len(out) >= num:
                    break
        if out:
            return out[:num]
    reps = (num + len(DEFAULT_PROMPTS) - 1) // len(DEFAULT_PROMPTS)
    return (DEFAULT_PROMPTS * reps)[:num]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output cache dir (must be new/empty)")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--jsonl", default="eval_datasets/gsm8k.jsonl")
    ap.add_argument("--num", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--layers", default="1,6,13,20,26", help="target_layer_ids (excl. final)")
    args = ap.parse_args()

    from mlx_lm import load

    model, tok = load(args.model)
    dims = model_dims(model)
    H = dims["hidden_size"]
    target_layer_ids = [int(x) for x in args.layers.split(",")]
    assert max(target_layer_ids) < dims["num_hidden_layers"] - 1, "layers must exclude final"
    print(f"model={args.model} hidden={H} layers={dims['num_hidden_layers']} "
          f"target_layer_ids={target_layer_ids}")

    prompts = load_prompts(args.jsonl, args.num)
    samples = []
    for i, text in enumerate(prompts):
        ids = tok.encode(text)[: args.max_length]
        if len(ids) < 2:
            continue
        input_ids = mx.array(ids, dtype=mx.int32)[None]  # [1, S]
        cap = capture_hidden_states(model, input_ids, target_layer_ids)
        S = input_ids.shape[1]
        sample = {
            "input_ids": input_ids[0],                                # [S]
            "loss_mask": mx.ones((S,), dtype=mx.uint8),               # supervise all
            "target_hidden_states": cap["target_hidden_states"][0],   # [S, L*H]
            "target_last_hidden_states": cap["target_last_hidden_states"][0],  # [S, H]
        }
        mx.eval(sample["target_hidden_states"], sample["target_last_hidden_states"])
        samples.append(sample)
        print(f"  [{i}] S={S}")

    manifest = write_target_cache(
        os.path.expanduser(args.out), samples,
        target_layer_ids=target_layer_ids, hidden_size=H,
        target_model_name_or_path=args.model,
        extra_manifest={"chat_template": None, "max_length": args.max_length, "regen": False},
    )
    print(f"\nwrote {manifest['num_samples']} samples to {args.out} "
          f"({manifest['shards'][0]['nbytes']/1e6:.2f} MB shard)")


if __name__ == "__main__":
    main()
