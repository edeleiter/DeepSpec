"""M3 round-trip test: write known arrays to the v2 cache, read them back exactly.

Validates the format helpers + writer + reader + the bf16 uint16 reinterpret, with
no model involved. int/uint must be bit-exact; bf16 must round-trip its stored
value exactly (bf16->bytes->bf16 is lossless).

Run:  python deepspec_mlx/tests/test_cache_reader_parity.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

import mlx.core as mx
import numpy as np

sys.path.insert(0, __file__.rsplit("/deepspec_mlx/", 1)[0])
from deepspec_mlx.data import CacheReader, write_target_cache


def build_samples(seed=0):
    rng = np.random.default_rng(seed)
    H, L = 16, 3               # tiny hidden, 3 target layers
    samples = []
    for S in (5, 8, 3):        # variable seq lengths across samples
        ids = mx.array(rng.integers(0, 1000, size=S).astype(np.int32))
        lm = mx.array((rng.random(S) > 0.3).astype(np.uint8))
        # bf16 values chosen so bf16 storage is lossless (exact in bf16)
        ths = mx.array(rng.integers(-8, 8, size=(S, L * H)).astype(np.float32)).astype(mx.bfloat16)
        tlhs = mx.array(rng.integers(-8, 8, size=(S, H)).astype(np.float32)).astype(mx.bfloat16)
        samples.append({
            "input_ids": ids, "loss_mask": lm,
            "target_hidden_states": ths, "target_last_hidden_states": tlhs,
        })
    return samples, H, L


def test_round_trip():
    samples, H, L = build_samples()
    tmp = tempfile.mkdtemp(prefix="dspark_cache_")
    cache_dir = os.path.join(tmp, "cache")
    try:
        write_target_cache(
            cache_dir, samples,
            target_layer_ids=[1, 6, 13], hidden_size=H,
            target_model_name_or_path="Qwen/Qwen3-0.6B",
        )
        reader = CacheReader(cache_dir)
        assert len(reader) == len(samples), f"len {len(reader)} != {len(samples)}"

        def eq(a, b):
            return np.array_equal(np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32)))

        for i, s in enumerate(samples):
            r = reader[i]
            assert tuple(r["input_ids"].shape) == tuple(s["input_ids"].shape)
            assert eq(r["input_ids"], s["input_ids"]), f"input_ids mismatch @ {i}"
            assert eq(r["loss_mask"], s["loss_mask"]), f"loss_mask mismatch @ {i}"
            assert tuple(r["target_hidden_states"].shape) == (s["input_ids"].shape[0], L * H)
            assert eq(r["target_hidden_states"], s["target_hidden_states"]), f"ths mismatch @ {i}"
            assert eq(r["target_last_hidden_states"], s["target_last_hidden_states"]), f"tlhs mismatch @ {i}"
            assert r["input_ids"].dtype == mx.int32
            assert r["target_hidden_states"].dtype == mx.bfloat16
            print(f"  sample {i}: S={s['input_ids'].shape[0]} OK "
                  f"(ids/loss/ths/tlhs bit-exact, dtypes correct)")
        reader.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return len(samples)


def main():
    print("== cache write -> read round-trip ==")
    try:
        n = test_round_trip()
        print(f"\nRESULT: PASS — {n} samples round-tripped bit-exact")
        return 0
    except AssertionError as e:
        print(f"\nRESULT: FAIL — {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
