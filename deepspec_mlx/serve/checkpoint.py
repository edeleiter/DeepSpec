"""Self-contained save/load for a trained DSpark draft.

A checkpoint dir holds:
  weights.safetensors  — all draft params (incl. the frozen embed/lm_head), so the
                         checkpoint fully reconstructs the draft without the target.
  draft.json           — the resolved DSparkDraftConfig + serving metadata (target_id,
                         arch, compute_dtype, model_id) so the server is generic: point
                         it at any checkpoint and it self-describes its target + runner.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict

import mlx.core as mx
from mlx.utils import tree_flatten

from deepspec_mlx.modeling.config import DSparkDraftConfig
from deepspec_mlx.modeling.dspark_qwen3 import Qwen3DSparkModel

_DTYPE_TO_STR = {mx.float32: "float32", mx.bfloat16: "bfloat16", mx.float16: "float16"}
_STR_TO_DTYPE = {v: k for k, v in _DTYPE_TO_STR.items()}


def save_draft(draft, out_dir, *, target_id, arch, model_id):
    """Persist a trained draft. arch in {'qwen3','qwen3_5'} selects the eval runner."""
    os.makedirs(out_dir, exist_ok=True)
    weights = dict(tree_flatten(draft.parameters()))
    mx.save_safetensors(os.path.join(out_dir, "weights.safetensors"), weights)
    cfg = asdict(draft.config)
    meta = {
        "config": cfg,
        "compute_dtype": _DTYPE_TO_STR[draft.compute_dtype],
        "target_id": str(target_id),
        "arch": str(arch),
        "model_id": str(model_id),
        # convenience duplicates the server reads directly
        "target_layer_ids": cfg["target_layer_ids"],
        "block_size": cfg["block_size"],
        "mask_token_id": cfg["mask_token_id"],
    }
    with open(os.path.join(out_dir, "draft.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def load_draft(out_dir):
    """Rebuild a draft from a checkpoint dir. Returns (draft, meta). No target needed."""
    out_dir = os.path.expanduser(out_dir)
    with open(os.path.join(out_dir, "draft.json"), encoding="utf-8") as f:
        meta = json.load(f)
    cfg = DSparkDraftConfig(**meta["config"])
    dtype = _STR_TO_DTYPE[meta["compute_dtype"]]
    draft = Qwen3DSparkModel(cfg, compute_dtype=dtype)
    draft.load_weights(os.path.join(out_dir, "weights.safetensors"))
    draft.assert_uniform_dtype()
    mx.eval(draft.parameters())
    return draft, meta
