"""MLX reader for the v2 target cache — mmap bytes -> mx.array.

Port of deepspec/data/target_cache_dataset.py:CacheDataset.__getitem__, torch-free.
bf16 tensors are read as raw uint16 and bit-reinterpreted via mx.array.view
(validated in the M1 bf16-bitcast probe). Returns per-sample dicts of mx.arrays.
"""

from __future__ import annotations

import json
import mmap
import os
from collections import OrderedDict

import mlx.core as mx
import numpy as np

from .cache_format import (
    INDEX_RECORD_SIZE,
    expected_target_cache_tensor_nbytes,
    unpack_index_record,
    validate_manifest,
)


class CacheReader:
    def __init__(self, cache_dir: str, max_open_shards: int = 4):
        self.cache_dir = os.path.abspath(cache_dir)
        with open(os.path.join(self.cache_dir, "manifest.json"), encoding="utf-8") as f:
            self.manifest = validate_manifest(json.load(f))
        self.hidden_size = int(self.manifest["hidden_size"])
        self.target_layer_ids = [int(x) for x in self.manifest["target_layer_ids"]]
        self.num_target_layers = len(self.target_layer_ids)
        self.num_samples = int(self.manifest["num_samples"])
        self._shard_files = {int(s["shard_id"]): s["file_name"] for s in self.manifest["shards"]}

        index_path = os.path.join(self.cache_dir, "samples.idx")
        self._index_fh = open(index_path, "rb")
        self._index_mmap = mmap.mmap(self._index_fh.fileno(), 0, access=mmap.ACCESS_READ)
        assert self._index_mmap.size() == self.num_samples * INDEX_RECORD_SIZE

        self._max_open = max_open_shards
        self._shards: "OrderedDict[int, mmap.mmap]" = OrderedDict()
        self._shard_fhs: dict = {}

    def __len__(self):
        return self.num_samples

    def _shard(self, shard_id: int) -> mmap.mmap:
        if shard_id in self._shards:
            self._shards.move_to_end(shard_id)
            return self._shards[shard_id]
        path = os.path.join(self.cache_dir, self._shard_files[shard_id])
        fh = open(path, "rb")
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        self._shard_fhs[shard_id] = fh
        self._shards[shard_id] = mm
        if len(self._shards) > self._max_open:
            old_id, old_mm = self._shards.popitem(last=False)
            old_mm.close()
            self._shard_fhs.pop(old_id).close()
        return mm

    def _read(self, mm, offset, count, np_dtype):
        arr = np.frombuffer(mm, dtype=np_dtype, count=int(count), offset=int(offset)).copy()
        return arr

    def __getitem__(self, index: int) -> dict:
        if not (0 <= index < self.num_samples):
            raise IndexError(index)
        rec = unpack_index_record(self._index_mmap, index * INDEX_RECORD_SIZE)
        assert int(rec["sample_id"]) == index, "index not dense/sorted by sample_id"
        S = int(rec["seq_len"])
        mm = self._shard(int(rec["shard_id"]))
        H, L = self.hidden_size, self.num_target_layers

        input_ids = mx.array(self._read(mm, rec["input_ids_offset"], S, np.int32))
        loss_mask = mx.array(self._read(mm, rec["loss_mask_offset"], S, np.uint8))
        ths = mx.array(self._read(mm, rec["target_hidden_states_offset"], S * L * H, np.uint16))
        ths = ths.view(mx.bfloat16).reshape(S, L * H)
        tlhs = mx.array(self._read(mm, rec["target_last_hidden_states_offset"], S * H, np.uint16))
        tlhs = tlhs.view(mx.bfloat16).reshape(S, H)

        return {
            "input_ids": input_ids,
            "loss_mask": loss_mask,
            "target_hidden_states": ths,
            "target_last_hidden_states": tlhs,
        }

    def close(self):
        self._index_mmap.close()
        self._index_fh.close()
        for mm in self._shards.values():
            mm.close()
        for fh in self._shard_fhs.values():
            fh.close()
        self._shards.clear()
        self._shard_fhs.clear()
