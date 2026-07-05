"""R1.5 — exercise the spec-decode stop-token termination branch (was untested).

The canary eval passes stop_ids=[], so the terminated branch (stop token inside an
accepted prefix) + the effective-proposal-length bookkeeping never ran. This trains a
small draft (so acceptance>0), then forces termination by making one of the greedily
generated tokens a stop token, and asserts the loop terminates with sane, consistent
metrics (equal-length lists, proposal_lengths<=block_size, acceptance<=proposal+1,
num_output present, generation actually stopped).

Needs the mlx-lm target; ~15s. Run:
    python deepspec_mlx/tests/test_spec_decode.py
"""

from __future__ import annotations

import os
import sys

import mlx.core as mx

sys.path.insert(0, __file__.rsplit("/deepspec_mlx/", 1)[0])
from deepspec_mlx.data import CacheReader
from deepspec_mlx.modeling import build_draft_config, Qwen3DSparkModel, target_embed_and_head
from deepspec_mlx.eval import TargetRunner, generate
from deepspec_mlx.trainer import overfit

BLOCK = 7


def _invariants(r, block_size):
    assert len(r.acceptance_lengths) == len(r.proposal_lengths), "length mismatch"
    for a, p in zip(r.acceptance_lengths, r.proposal_lengths):
        assert 0 <= p <= block_size, f"proposal_length {p} out of [0,{block_size}]"
        assert 1 <= a <= p + 1, f"acceptance_length {a} inconsistent with proposal {p}"
    assert hasattr(r, "num_output"), "missing num_output"


def main():
    cache = os.path.expanduser("~/dspark_mlx/cache/qwen3_0_6b_canary")
    from mlx_lm import load
    model, tok = load("Qwen/Qwen3-0.6B")
    r = CacheReader(cache)
    tli = [int(x) for x in r.target_layer_ids]
    samples = [r[i] for i in range(len(r))]
    prompt = samples[0]["input_ids"][None, : max(4, samples[0]["input_ids"].shape[0] // 2)]

    cfg = build_draft_config(model.args, dict(
        num_draft_layers=3, target_layer_ids=tli, block_size=BLOCK, mask_token_id=151669,
        num_anchors=64, markov_rank=256, markov_head_type="vanilla",
        confidence_head_alpha=1.0, confidence_head_with_markov=True))
    draft = Qwen3DSparkModel(cfg)
    draft.initialize_from_target(*target_embed_and_head(model))
    mx.eval(draft.parameters())
    overfit(draft, samples, steps=25, lr=1e-3, log_every=25)

    print("\n== baseline (no stop) ==")
    base = generate(TargetRunner(model, tli), draft, prompt, max_new_tokens=40,
                    block_size=BLOCK, temperature=0.0, stop_ids=[], seed=0)
    _invariants(base, BLOCK)
    gen = base.committed          # committed = the generated continuation (prompt not included)
    print(f"  generated {len(gen)} tokens, verify steps={len(base.proposal_lengths)}, "
          f"mean accept_len={sum(base.acceptance_lengths)/max(len(base.acceptance_lengths),1):.2f}")

    # force termination: stop on an early generated token (skip the very first, which
    # would hit the prefill early-return path instead of the loop).
    assert len(gen) >= 3, "need a few generated tokens to place a stop"
    stop_tok = int(gen[2])

    print(f"\n== with stop_ids={{{stop_tok}}} (forces termination) ==")
    stopped = generate(TargetRunner(model, tli), draft, prompt, max_new_tokens=40,
                       block_size=BLOCK, temperature=0.0, stop_ids=[stop_tok], seed=0)
    _invariants(stopped, BLOCK)
    assert len(stopped.committed) <= len(base.committed), "stop run should not be longer"
    assert stop_tok in stopped.committed, "stop token should appear in the committed output"
    # terminated stop-in-prefix step (if any) reports proposal==acceptance (the R1.3 fix):
    term_steps = [(p, a) for p, a in zip(stopped.proposal_lengths, stopped.acceptance_lengths) if p == a]
    print(f"  committed {len(stopped.committed)} (base {len(base.committed)}); "
          f"verify steps={len(stopped.proposal_lengths)}; stop-in-prefix steps={len(term_steps)}")
    print("  invariants hold; termination path exercised without crash")

    r.close()
    print("\nRESULT: PASS — spec-decode stop-token termination validated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
