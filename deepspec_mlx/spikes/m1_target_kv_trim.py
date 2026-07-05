"""M1 spike #1 — target KV-cache trim (rewind) round-trip.

THE question this answers: does mlx-lm's prompt cache support the `.crop(start)`
rewind that the DSpark verifier depends on (base_evaluator.py:418/:425)? If the
target cache can be trimmed back to an earlier length and continue with IDENTICAL
logits, the target side of the spec-decode loop is unblocked.

This is a throwaway diagnostic: it introspects the mlx-lm cache API (which has
churned across releases) and then runs a correctness test:

    Reference : one forward over the full N-token sequence  -> logits_full
    Rewound   : forward N-k tokens, forward k more (append), TRIM back k,
                forward the last k again                     -> logits_rewound
    Assert    : logits_full[:, -1] == logits_rewound[:, -1]  (bf16-exact)

Run (after installing mlx + mlx-lm in the venv):
    python deepspec_mlx/spikes/m1_target_kv_trim.py
    python deepspec_mlx/spikes/m1_target_kv_trim.py --model Qwen/Qwen3-0.6B --k 4
"""

from __future__ import annotations

import argparse
import sys


def introspect_cache_api() -> dict:
    """Print what the installed mlx-lm exposes for prompt caching."""
    import mlx
    import mlx_lm

    print(f"  mlx        {getattr(mlx, '__version__', '?')}")
    print(f"  mlx_lm     {getattr(mlx_lm, '__version__', '?')}")

    from mlx_lm.models import cache as cache_mod

    names = [n for n in dir(cache_mod) if not n.startswith("_")]
    print(f"  mlx_lm.models.cache exports: {names}")

    api = {}
    for fn in ("make_prompt_cache", "trim_prompt_cache"):
        api[fn] = getattr(cache_mod, fn, None)
        print(f"    {fn}: {'present' if api[fn] else 'MISSING'}")

    kv = getattr(cache_mod, "KVCache", None)
    api["KVCache"] = kv
    if kv is not None:
        methods = [m for m in dir(kv) if not m.startswith("__")]
        print(f"    KVCache methods: {methods}")
        for m in ("trim", "is_trimmable"):
            print(f"      KVCache.{m}: {'present' if hasattr(kv, m) else 'MISSING'}")
    return api


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--prompt", default="The capital of France is Paris, and the capital of Japan is")
    ap.add_argument("--k", type=int, default=4, help="tokens to append then rewind")
    args = ap.parse_args()

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

    print("== mlx-lm cache API ==")
    introspect_cache_api()

    print(f"\n== loading {args.model} ==")
    model, tok = load(args.model)

    import numpy as np

    ids = tok.encode(args.prompt)
    N = len(ids)
    k = min(args.k, N - 1)
    assert k >= 1, "need at least 1 token to rewind"
    full = mx.array(ids)[None]  # [1, N]
    prefix = full[:, : N - k]   # [1, N-k]
    tail = full[:, N - k :]     # [1, k]
    # Junk tokens to append then trim — DIFFERENT from the tail, so a correct trim
    # must fully erase them (reversed tail; distinct as long as the tail isn't a
    # palindrome, which is vanishingly unlikely for real text).
    junk = tail[:, ::-1]
    print(f"  seq len N={N}, rewind k={k} (prefix={N-k}, tail={k})")

    def last_logits(logits):
        v = logits[0, -1]
        mx.eval(v)
        return np.array(v.astype(mx.float32))

    # (A) single forward over the whole sequence — the "chunking noise" reference.
    cache_a = make_prompt_cache(model)
    L_full = last_logits(model(full, cache=cache_a))

    # (B) chunked, NO trim: prefix then tail. Same accumulation order as (C).
    cache_b = make_prompt_cache(model)
    model(prefix, cache=cache_b)
    L_chunked = last_logits(model(tail, cache=cache_b))

    # (C) chunked WITH rewind: prefix, append JUNK, trim k, then tail.
    cache_c = make_prompt_cache(model)
    model(prefix, cache=cache_c)
    off_prefix = cache_c[0].offset
    model(junk, cache=cache_c)              # speculative append (rejected block)
    off_appended = cache_c[0].offset
    trimmed = trim_prompt_cache(cache_c, k)  # rewind exactly k tokens
    off_trim = cache_c[0].offset
    L_trimmed = last_logits(model(tail, cache=cache_c))

    print("\n== trim bookkeeping ==")
    print(f"  offset: prefix={off_prefix} (exp {N-k}), appended={off_appended} (exp {N}), "
          f"after trim={off_trim} (exp {N-k}); trim_prompt_cache returned {trimmed} (exp {k})")

    # The decisive test: append-then-trim must leave the cache BIT-IDENTICAL to
    # never-having-appended, so (C) == (B) exactly. (A) vs (B) is just bf16
    # chunking noise and is reported only for context.
    d_chunk_noise = float(np.max(np.abs(L_full - L_chunked)))
    d_trim = float(np.max(np.abs(L_chunked - L_trimmed)))
    print("\n== correctness ==")
    print(f"  [context] full(1x) vs chunked(no-trim) max|Δ|  : {d_chunk_noise:.6g}  (pure bf16 chunk noise)")
    print(f"  [DECISIVE] chunked vs append+trim   max|Δ|     : {d_trim:.6g}  (must be ~0)")
    print(f"  argmax matches (all three): "
          f"{int(np.argmax(L_full))==int(np.argmax(L_chunked))==int(np.argmax(L_trimmed))}")

    bookkeeping_ok = (off_prefix == N - k and off_appended == N and off_trim == N - k and trimmed == k)
    ok = bookkeeping_ok and d_trim < 1e-3
    print(f"\nRESULT: {'PASS — target KV trim/rewind leaves no trace' if ok else 'FAIL — investigate above'}")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ImportError as e:
        print(f"[import error] {e}\n\nInstall the MLX runtime first: uv sync --project deepspec_mlx", file=sys.stderr)
        sys.exit(2)
