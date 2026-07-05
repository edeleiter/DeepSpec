"""MLX writer for the v2 target cache — mx.array -> on-disk bytes.

Single-shard, single-process writer (enough for the canary; the torch reference's
async/DDP/multi-shard writer is not needed on one Mac). Produces shard-00000.bin,
samples.idx, manifest.json in the canonical protocol so CacheReader (and the torch
CacheDataset) can read it back byte-for-byte.
"""

from __future__ import annotations

import json
import os

import mlx.core as mx
import numpy as np

from .cache_format import (
    INDEX_RECORD_SIZE,
    TARGET_CACHE_HIDDEN_DTYPE,
    TARGET_CACHE_MASK_DTYPE,
    TARGET_CACHE_TOKEN_DTYPE,
    TARGET_CACHE_VERSION,
    pack_index_record,
)


def _i32_bytes(a: mx.array) -> bytes:
    return np.array(a.astype(mx.int32)).astype("<i4").tobytes()


def _u8_bytes(a: mx.array) -> bytes:
    return np.array(a.astype(mx.uint8)).astype(np.uint8).tobytes()


def _bf16_bytes(a: mx.array) -> bytes:
    # bit-reinterpret bf16 -> uint16 raw bytes (little-endian)
    return np.array(a.astype(mx.bfloat16).view(mx.uint16)).astype("<u2").tobytes()


def write_target_cache(
    cache_dir: str,
    samples: list,
    *,
    target_layer_ids,
    hidden_size: int,
    target_model_name_or_path: str,
    extra_manifest: dict | None = None,
):
    """Write a list of sample dicts to a fresh cache_dir.

    Each sample dict needs: input_ids [S] (int), loss_mask [S] (0/1),
    target_hidden_states [S, L*H] (bf16), target_last_hidden_states [S, H] (bf16).
    attention_mask is written as all-ones (canary has no padding).
    """
    cache_dir = os.path.abspath(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    if os.listdir(cache_dir):
        raise FileExistsError(f"cache dir not empty: {cache_dir}")

    target_layer_ids = [int(x) for x in target_layer_ids]
    L = len(target_layer_ids)
    shard_name = "shard-00000.bin"
    shard_path = os.path.join(cache_dir, shard_name)
    index_records = []

    with open(shard_path, "wb") as shard:
        cursor = 0
        for sid, s in enumerate(samples):
            S = int(s["input_ids"].shape[0])
            assert tuple(s["target_hidden_states"].shape) == (S, L * hidden_size), \
                f"sample {sid}: target_hidden_states {tuple(s['target_hidden_states'].shape)} != {(S, L*hidden_size)}"
            assert tuple(s["target_last_hidden_states"].shape) == (S, hidden_size)

            attn = mx.ones((S,), dtype=mx.uint8)
            blobs = [
                ("input_ids", _i32_bytes(s["input_ids"])),
                ("attention_mask", _u8_bytes(attn)),
                ("loss_mask", _u8_bytes(s["loss_mask"])),
                ("target_hidden_states", _bf16_bytes(s["target_hidden_states"])),
                ("target_last_hidden_states", _bf16_bytes(s["target_last_hidden_states"])),
            ]
            offsets = {}
            for name, b in blobs:
                offsets[name] = cursor
                shard.write(b)
                cursor += len(b)

            index_records.append(pack_index_record(
                sample_id=sid, shard_id=0, seq_len=S,
                input_ids_offset=offsets["input_ids"],
                attention_mask_offset=offsets["attention_mask"],
                loss_mask_offset=offsets["loss_mask"],
                target_hidden_states_offset=offsets["target_hidden_states"],
                target_last_hidden_states_offset=offsets["target_last_hidden_states"],
            ))
        shard_bytes = cursor

    with open(os.path.join(cache_dir, "samples.idx"), "wb") as f:
        for rec in index_records:
            f.write(rec)

    manifest = {
        "version": TARGET_CACHE_VERSION,
        "num_samples": len(samples),
        "num_shards": 1,
        "target_layer_ids": target_layer_ids,
        "hidden_dtype": TARGET_CACHE_HIDDEN_DTYPE,
        "token_dtype": TARGET_CACHE_TOKEN_DTYPE,
        "mask_dtype": TARGET_CACHE_MASK_DTYPE,
        "index_record_size": INDEX_RECORD_SIZE,
        "hidden_size": int(hidden_size),
        "target_model_name_or_path": str(target_model_name_or_path),
        "shards": [{"shard_id": 0, "file_name": "shard-00000.bin", "nbytes": shard_bytes}],
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    with open(os.path.join(cache_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return manifest
