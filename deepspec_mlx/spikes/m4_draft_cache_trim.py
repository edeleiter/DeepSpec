"""M1 spike #4 — draft KV-cache: mixed-append-then-trim leaves no trace.

The DSpark draft attention (modeling.py:104) builds keys/values as
`cat([k_ctx, k_noise])` and calls `past_key_values.update(...)` on the whole thing,
then `forward_dspark_draft_block` (draft_ops.py:44) does `crop(start)` to drop the
speculative noise block. Because the NOISE keys are appended LAST, the draft-cache
rewind is exactly `append(cat([ctx, noise]))` then `trim(len(noise))`.

This spike proves that pattern is byte-exact on mlx-lm's KVCache:

    cache_A: append(ctx1); append(ctx2)                       # noise never added
    cache_B: append(ctx1); append(cat([ctx2, noise])); trim(len(noise))
    assert cache_A.keys == cache_B.keys and .values equal, offsets equal

If A == B exactly, the draft cache needs NO bespoke class — the standard KVCache +
trim handles the rewind, and the "custom" part is purely the attention math (M4).
(The plan's fallback — a cache-free O(n^2) draft — is therefore not needed.)

Run:
    python deepspec_mlx/spikes/m4_draft_cache_trim.py
"""

from __future__ import annotations

import sys


def fetch(cache):
    """Return (keys, values) currently held, trimmed to offset."""
    import mlx.core as mx
    # update_and_fetch with a zero-width tensor would be awkward; use .state,
    # which KVCache exposes as (keys, values) sliced to offset.
    keys, values = cache.state
    mx.eval(keys, values)
    return keys, values


def main() -> int:
    import mlx.core as mx
    import numpy as np
    from mlx_lm.models.cache import KVCache, trim_prompt_cache, make_prompt_cache  # noqa: F401

    B, Hkv, d = 1, 8, 128
    n_ctx1, n_ctx2, n_noise = 5, 3, 7  # accepted-context grows; noise = block_size

    def rk(n):  # random keys/values [B,Hkv,n,d]
        return mx.random.normal((B, Hkv, n, d)).astype(mx.bfloat16)

    ctx1_k, ctx1_v = rk(n_ctx1), rk(n_ctx1)
    ctx2_k, ctx2_v = rk(n_ctx2), rk(n_ctx2)
    noise_k, noise_v = rk(n_noise), rk(n_noise)

    # cache_A: only the committed context ever enters.
    A = KVCache()
    A.update_and_fetch(ctx1_k, ctx1_v)
    A.update_and_fetch(ctx2_k, ctx2_v)

    # cache_B: context1, then cat([context2, noise]) in ONE update, then trim noise.
    B_ = KVCache()
    B_.update_and_fetch(ctx1_k, ctx1_v)
    B_.update_and_fetch(
        mx.concatenate([ctx2_k, noise_k], axis=2),
        mx.concatenate([ctx2_v, noise_v], axis=2),
    )
    off_before = B_.offset
    trimmed = B_.trim(n_noise)
    off_after = B_.offset

    ka, va = fetch(A)
    kb, vb = fetch(B_)

    print("== offsets ==")
    print(f"  A.offset            = {A.offset}  (expect {n_ctx1 + n_ctx2})")
    print(f"  B before trim       = {off_before}  (expect {n_ctx1 + n_ctx2 + n_noise})")
    print(f"  B.trim() returned   = {trimmed}  (expect {n_noise})")
    print(f"  B.offset after trim = {off_after}  (expect {n_ctx1 + n_ctx2})")

    def maxdiff(x, y):
        return float(np.max(np.abs(np.array(x.astype(mx.float32)) - np.array(y.astype(mx.float32)))))

    print("\n== state equality (A == B after trim) ==")
    print(f"  keys   shape A{tuple(ka.shape)} B{tuple(kb.shape)}")
    dk = maxdiff(ka, kb) if ka.shape == kb.shape else float("nan")
    dv = maxdiff(va, vb) if va.shape == vb.shape else float("nan")
    print(f"  max|Δ| keys   = {dk}")
    print(f"  max|Δ| values = {dv}")

    ok = (
        A.offset == n_ctx1 + n_ctx2
        and off_after == n_ctx1 + n_ctx2
        and trimmed == n_noise
        and ka.shape == kb.shape
        and dk == 0.0
        and dv == 0.0
    )
    print(f"\nRESULT: {'PASS — draft cache = standard KVCache + trim (no bespoke class needed)' if ok else 'FAIL — investigate'}")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ImportError as e:
        print(f"[import error] {e}", file=sys.stderr)
        sys.exit(2)
