"""Target-cache on-disk protocol (v2) — pure-python, torch-free copy.

Byte-for-byte identical to deepspec/data/target_cache_dataset.py's protocol so a
cache written here is readable by the torch reference and vice-versa. Only the
format constants + index/manifest helpers live here; the MLX reader/writer that
turn bytes into mx.arrays are in cache_reader.py / cache_writer.py.

Per-sample shard layout (canonical write order), all little-endian:
    input_ids                 int32   [seq]
    attention_mask            uint8   [seq]
    loss_mask                 uint8   [seq]
    target_hidden_states      bfloat16[seq, num_target_layers*hidden]   (raw uint16)
    target_last_hidden_states bfloat16[seq, hidden]                     (raw uint16)

Index record (samples.idx), struct "<QIIQQQQQ":
    sample_id(Q) shard_id(I) seq_len(I) + 5 byte-offsets(Q) into the shard.
"""

from __future__ import annotations

import struct

TARGET_CACHE_VERSION = 2
INDEX_RECORD_STRUCT = struct.Struct("<QIIQQQQQ")
INDEX_RECORD_SIZE = INDEX_RECORD_STRUCT.size  # 56

TARGET_CACHE_HIDDEN_DTYPE = "bfloat16"
TARGET_CACHE_TOKEN_DTYPE = "int32"
TARGET_CACHE_MASK_DTYPE = "uint8"


def expected_target_cache_tensor_nbytes(*, seq_len, hidden_size, num_target_layers):
    seq_len, hidden_size, num_target_layers = int(seq_len), int(hidden_size), int(num_target_layers)
    return {
        "input_ids": seq_len * 4,
        "attention_mask": seq_len,
        "loss_mask": seq_len,
        "target_hidden_states": seq_len * num_target_layers * hidden_size * 2,
        "target_last_hidden_states": seq_len * hidden_size * 2,
    }


def pack_index_record(
    *, sample_id, shard_id, seq_len,
    input_ids_offset, attention_mask_offset, loss_mask_offset,
    target_hidden_states_offset, target_last_hidden_states_offset,
):
    return INDEX_RECORD_STRUCT.pack(
        int(sample_id), int(shard_id), int(seq_len),
        int(input_ids_offset), int(attention_mask_offset), int(loss_mask_offset),
        int(target_hidden_states_offset), int(target_last_hidden_states_offset),
    )


def unpack_index_record(buffer, offset: int = 0):
    (sample_id, shard_id, seq_len, io, ao, lo, tho, tlo) = INDEX_RECORD_STRUCT.unpack_from(buffer, offset)
    return {
        "sample_id": sample_id, "shard_id": shard_id, "seq_len": seq_len,
        "input_ids_offset": io, "attention_mask_offset": ao, "loss_mask_offset": lo,
        "target_hidden_states_offset": tho, "target_last_hidden_states_offset": tlo,
    }


def validate_manifest(manifest):
    required = {
        "version", "num_samples", "num_shards", "target_layer_ids", "hidden_dtype",
        "token_dtype", "mask_dtype", "index_record_size", "hidden_size", "shards",
    }
    missing = sorted(required - set(manifest))
    assert not missing, f"manifest missing fields {missing}"
    assert int(manifest["version"]) == TARGET_CACHE_VERSION, f"bad version {manifest['version']}"
    assert manifest["hidden_dtype"] == TARGET_CACHE_HIDDEN_DTYPE
    assert manifest["token_dtype"] == TARGET_CACHE_TOKEN_DTYPE
    assert manifest["mask_dtype"] == TARGET_CACHE_MASK_DTYPE
    assert int(manifest["index_record_size"]) == INDEX_RECORD_SIZE
    lids = [int(x) for x in manifest["target_layer_ids"]]
    assert lids and lids == sorted(lids), "target_layer_ids must be sorted, non-empty"
    return manifest
