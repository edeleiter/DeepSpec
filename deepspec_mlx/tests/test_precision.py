"""R1.5 — cover the precision path that actually runs in production.

The reviewers' #1 finding: every other test builds a pure-fp32 model and skips
initialize_from_target, so the real path (bf16 target heads copied into the draft,
scheme C) was untested. This exercises it in BOTH compute modes and checks the
assert_uniform_dtype guardrail catches a hybrid.

Run:  python deepspec_mlx/tests/test_precision.py
"""

from __future__ import annotations

import sys

import mlx.core as mx
import numpy as np

sys.path.insert(0, __file__.rsplit("/deepspec_mlx/", 1)[0])
from deepspec_mlx.modeling.dspark_qwen3 import Qwen3DSparkModel
from deepspec_mlx.modeling.loss import compute_dspark_loss
from deepspec_mlx.tests.test_dspark_forward import tiny_config, make_inputs, LG


def _bf16_head(cfg, seed):
    rng = np.random.default_rng(seed)
    return mx.array(rng.standard_normal((cfg.vocab_size, cfg.hidden_size)).astype(np.float32)).astype(mx.bfloat16)


def test_initialize_from_target_both_modes():
    cfg = tiny_config()
    for dt in (mx.float32, mx.bfloat16):
        m = Qwen3DSparkModel(cfg, compute_dtype=dt)
        embed_w = _bf16_head(cfg, 0)          # untied: distinct embed and head, both bf16
        head_w = _bf16_head(cfg, 1)
        m.initialize_from_target(embed_w, head_w)
        # heads cast to compute dtype (NOT left at the target's bf16)
        assert m.embed_tokens.weight.dtype == dt, (dt, m.embed_tokens.weight.dtype)
        assert m.lm_head.weight.dtype == dt
        # untied: the two heads are distinct arrays with distinct values
        assert not np.array_equal(np.array(m.embed_tokens.weight.astype(mx.float32)),
                                  np.array(m.lm_head.weight.astype(mx.float32))), "heads must not alias"
        m.assert_uniform_dtype()              # guardrail passes
        # the real forward+loss path runs and is finite
        input_ids, loss_mask, ths, tlhs, anchors, keep = make_inputs(cfg)
        out = m(input_ids, ths, loss_mask, tlhs, anchor_positions=anchors, block_keep_mask=keep)
        loss, _ = compute_dspark_loss(out, **LG)
        mx.eval(loss)
        assert np.isfinite(float(loss)), f"loss not finite in {dt} mode"
        print(f"  {str(dt):16s}: heads cast OK, distinct, forward+loss finite (loss={float(loss):.3f})")
    return True


def test_guardrail_catches_hybrid():
    cfg = tiny_config()
    m = Qwen3DSparkModel(cfg, compute_dtype=mx.float32)
    # force a hybrid: leave a head at bf16
    m.embed_tokens.weight = _bf16_head(cfg, 2)   # bf16 into an fp32 model
    try:
        m.assert_uniform_dtype()
    except AssertionError:
        print("  assert_uniform_dtype correctly rejected a bf16-head-in-fp32-model hybrid")
        return True
    raise AssertionError("guardrail failed to catch the hybrid")


def main():
    tests = [
        ("initialize_from_target both modes", test_initialize_from_target_both_modes),
        ("guardrail catches hybrid", test_guardrail_catches_hybrid),
    ]
    failed = 0
    for name, fn in tests:
        print(f"\n== {name} ==")
        try:
            fn()
            print("  PASS")
        except AssertionError as e:
            print(f"  FAIL: {e}")
            failed += 1
    print(f"\nRESULT: {'PASS — precision scheme C validated' if failed == 0 else f'FAIL ({failed})'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
