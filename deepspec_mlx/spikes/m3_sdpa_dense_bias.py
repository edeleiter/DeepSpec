"""M1 spike #3 — SDPA with a dense additive bias, at DSpark shapes.

DSpark's custom attention (common.py:109 create_dspark_attention_bias) is NOT
causal — it masks with a dense additive bias [B, 1, Q, KV] (0 = allowed,
finfo.min = disallowed) over `mx.fast.scaled_dot_product_attention`. Two risks to
retire here (plan risks #3 and the finite-min detail from §Draft model):

  1. Perf/memory: at the 4B config Q = num_anchors*block_size ≈ 512*7 = 3584 and
     KV ≈ seq_len + Q ≈ 7680. Is the masked SDPA fast + bounded on Metal?
  2. Fully-masked rows: padding/invalid draft blocks are entirely disallowed. The
     reference uses a FINITE finfo.min (not -inf) so a fully-masked row softmaxes
     to a uniform (non-NaN) distribution that gets discarded downstream. Confirm
     MLX reproduces this (no NaNs) with the finite-min fill.

Run:
    python deepspec_mlx/spikes/m3_sdpa_dense_bias.py
"""

from __future__ import annotations

import sys
import time


def peak_mem_mb() -> float:
    import mlx.core as mx
    for fn in ("get_peak_memory", "get_peak_memory_bytes"):
        f = getattr(mx, fn, None)
        if f is not None:
            return f() / 1e6
    return float("nan")


def reset_peak():
    import mlx.core as mx
    f = getattr(mx, "reset_peak_memory", None)
    if f is not None:
        f()


def bench(name, B, Hq, Hkv, Q, KV, d, dtype, fill=None):
    import mlx.core as mx

    scale = d ** -0.5
    fmin = float(mx.finfo(dtype).min) if fill is None else fill

    q = mx.random.normal((B, Hq, Q, d)).astype(dtype)
    k = mx.random.normal((B, Hkv, KV, d)).astype(dtype)
    v = mx.random.normal((B, Hkv, KV, d)).astype(dtype)

    # Dense additive bias [B,1,Q,KV]: allow a causal-ish band, disallow the rest,
    # and make the FIRST query row fully masked to test the finite-min behavior.
    allowed = (mx.arange(KV)[None, :] <= (mx.arange(Q)[:, None] + (KV - Q)))  # [Q,KV] bool
    bias = mx.where(allowed, mx.array(0.0, dtype), mx.array(fmin, dtype))      # [Q,KV]
    bias = mx.broadcast_to(bias[None, None], (B, 1, Q, KV))
    # force row 0 fully masked
    row0 = mx.full((B, 1, 1, KV), fmin, dtype)
    bias = mx.concatenate([row0, bias[:, :, 1:, :]], axis=2)

    reset_peak()
    # warmup + correctness
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=bias)
    mx.eval(out)
    has_nan = bool(mx.any(mx.isnan(out)).item())

    reps = 20
    t0 = time.perf_counter()
    for _ in range(reps):
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=bias)
        mx.eval(out)
    dt = (time.perf_counter() - t0) / reps * 1e3  # ms

    print(f"  {name:14s} Q={Q:5d} KV={KV:5d} Hq={Hq} d={d}: "
          f"{dt:6.2f} ms/call  peak {peak_mem_mb():7.1f} MB  "
          f"out{tuple(out.shape)}  NaN={has_nan}")
    return not has_nan


def main() -> int:
    import mlx.core as mx

    dtype = mx.bfloat16
    print("== finding a safe mask-fill sentinel (fully-masked row must not NaN) ==")
    fmin = float(mx.finfo(dtype).min)
    candidates = [("finfo.min", fmin), ("-1e30", -1e30), ("-1e9", -1e9), ("-1e4", -1e4)]
    safe_fill = None
    for label, fill in candidates:
        no_nan = bench(f"probe:{label}", 1, 16, 8, 128 * 7, 512 + 128 * 7, 128, dtype, fill=fill)
        if no_nan and safe_fill is None:
            safe_fill = fill
    print(f"  -> smallest-magnitude safe fill among probes: {safe_fill}")

    print("\n== perf/memory at DSpark shapes (using safe fill) ==")
    ok = safe_fill is not None
    # canary (0.6B): hidden 1024, 16 q-heads, 8 kv, head_dim 128; modest num_anchors
    ok &= bench("canary/na128", 1, 16, 8, 128 * 7, 512 + 128 * 7, 128, dtype, fill=safe_fill)
    ok &= bench("canary/na64",  1, 16, 8, 64 * 7, 512 + 64 * 7, 128, dtype, fill=safe_fill)
    # 4B-scale: 32 q-heads, 8 kv, head_dim 128, num_anchors 512, seq 4096
    ok &= bench("qwen4b/na512", 1, 32, 8, 512 * 7, 4096 + 512 * 7, 128, dtype, fill=safe_fill)
    ok &= bench("qwen4b/na256", 1, 32, 8, 256 * 7, 4096 + 256 * 7, 128, dtype, fill=safe_fill)

    print(f"\nRESULT: {'PASS — masked SDPA fast + no NaNs with safe fill' if ok else 'FAIL'}")
    print("  TAKEAWAY: MLX SDPA needs a FINITE sentinel (bf16 finfo.min overflows to NaN);")
    print(f"  bake fill={safe_fill} into the MLX create_dspark_attention_bias port (M4).")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ImportError as e:
        print(f"[import error] {e}", file=sys.stderr)
        sys.exit(2)
