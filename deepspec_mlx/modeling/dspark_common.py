"""DSpark index/sampler/bias ops in MLX — port of deepspec/modeling/dspark/common.py.

Pure array algebra (no learned params here). The RNG in sample_anchor_positions
differs from torch, so tests inject fixed anchors. Key MLX-specific choice: the
dense attention bias uses a FINITE -1e9 sentinel, NOT finfo.min (MLX SDPA NaNs on
fully-masked rows — established in the M1 SDPA spike).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx

# Finite mask fill (see deepspec_mlx/spikes/m3_sdpa_dense_bias.py): bf16 finfo.min
# overflows to NaN inside mx.fast.scaled_dot_product_attention on fully-masked rows.
MASK_FILL = -1e9


@dataclass
class DSparkForwardOutput:
    draft_logits: mx.array            # [B, num_anchors, block_size, vocab]
    target_ids: mx.array              # [B, num_anchors, block_size]
    eval_mask: mx.array               # [B, num_anchors, block_size] bool
    block_keep_mask: mx.array         # [B, num_anchors] bool
    confidence_pred: Optional[mx.array] = None       # [B, num_anchors, block_size]
    aligned_target_logits: Optional[mx.array] = None  # [B, num_anchors, block_size, vocab]


def rotate_half(x: mx.array) -> mx.array:
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2:]
    return mx.concatenate([-x2, x1], axis=-1)


def rope_cos_sin(position_ids: mx.array, head_dim: int, theta: float, dtype=mx.float32):
    """cos/sin tables for explicit RoPE application (NeoX/HF convention).

    position_ids: [B, L] -> cos, sin each [B, L, head_dim]. Computed in fp32 then
    cast, matching Qwen3RotaryEmbedding (attention_scaling=1.0 for the default rope).
    """
    half = head_dim // 2
    inv_freq = theta ** (-mx.arange(0, half, dtype=mx.float32) * 2.0 / head_dim)  # [half]
    pos = position_ids.astype(mx.float32)                       # [B, L]
    freqs = pos[..., None] * inv_freq[None, None, :]            # [B, L, half]
    emb = mx.concatenate([freqs, freqs], axis=-1)               # [B, L, head_dim]
    return mx.cos(emb).astype(dtype), mx.sin(emb).astype(dtype)


def apply_rope_q(q: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """q: [B, H, q_len, d]; cos/sin: [B, KV, d]. q uses the LAST q_len slots."""
    q_len = q.shape[-2]
    cq = cos[:, -q_len:, :][:, None, :, :]   # [B,1,q_len,d]
    sq = sin[:, -q_len:, :][:, None, :, :]
    return q * cq + rotate_half(q) * sq


def apply_rope_k(k: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """k: [B, H, KV, d]; cos/sin: [B, KV, d]. k uses ALL slots."""
    ck = cos[:, None, :, :]
    sk = sin[:, None, :, :]
    return k * ck + rotate_half(k) * sk


def build_anchor_candidate_mask(seq_len: int, loss_mask: mx.array) -> mx.array:
    nc = max(seq_len - 1, 0)
    if nc == 0:
        return loss_mask[:, :0] > 0.5
    anchor_valid = loss_mask[:, :nc] > 0.5
    first_target_valid = loss_mask[:, 1: nc + 1] > 0.5
    return anchor_valid & first_target_valid


def sample_anchor_positions(seq_len, loss_mask, num_anchors, key=None):
    """Random subset of valid anchor positions. Returns (anchors[B,na] int32,
    block_keep_mask[B,na] bool). RNG differs from torch; inject for parity tests."""
    valid = build_anchor_candidate_mask(seq_len, loss_mask)   # [B, nc]
    B = loss_mask.shape[0]
    nc = valid.shape[1]
    max_n = int(num_anchors)
    if nc == 0:
        return mx.zeros((B, max_n), dtype=mx.int32), mx.zeros((B, max_n), dtype=mx.bool_)

    if key is None:
        raise ValueError(
            "sample_anchor_positions requires an explicit RNG key; a default key would "
            "draw identical anchors every call. Pass key=mx.random.split(...)."
        )
    valid_counts = valid.sum(axis=1)                          # [B]
    indices = mx.broadcast_to(mx.arange(nc, dtype=mx.int32)[None, :], (B, nc))
    masked_indices = mx.where(valid, indices, mx.array(seq_len + 1, mx.int32))
    rv = mx.random.uniform(shape=(B, nc), key=key)
    rv = mx.where(valid, rv, mx.array(2.0))
    sorted_idx = mx.argsort(rv, axis=1)
    gathered = mx.take_along_axis(masked_indices, sorted_idx, axis=1)
    if nc < max_n:
        pad = mx.full((B, max_n - nc), seq_len + 1, dtype=gathered.dtype)
        gathered = mx.concatenate([gathered, pad], axis=1)
    anchors = mx.sort(gathered[:, :max_n], axis=1)
    keep = mx.arange(max_n)[None, :] < mx.minimum(valid_counts, max_n)[:, None]
    anchors = mx.where(keep, anchors, mx.zeros_like(anchors)).astype(mx.int32)
    return anchors, keep


def create_position_ids(anchor_positions: mx.array, block_size: int) -> mx.array:
    B, num_blocks = anchor_positions.shape
    offsets = mx.arange(block_size, dtype=anchor_positions.dtype)[None, None, :]
    return (anchor_positions[:, :, None] + offsets).reshape(B, num_blocks * block_size)


def create_noise_ids(input_ids, anchor_positions, block_keep_mask, *, mask_token_id, block_size):
    """Noise/draft token ids [B, num_blocks*block_size]: each block is
    [anchor_token, mask, mask, ...] (mask when the block is not kept). Built via
    reshape (the anchor is always the block's first slot) — no scatter needed."""
    B, num_blocks = anchor_positions.shape
    anchor_tokens = mx.take_along_axis(input_ids, anchor_positions, axis=1)  # [B, num_blocks]
    first = mx.where(block_keep_mask, anchor_tokens, mx.array(mask_token_id, input_ids.dtype))
    rest = mx.full((B, num_blocks, block_size - 1), mask_token_id, dtype=input_ids.dtype)
    noise_3d = mx.concatenate([first[:, :, None], rest], axis=2)  # [B, num_blocks, block_size]
    return noise_3d.reshape(B, num_blocks * block_size)


def create_dspark_attention_bias(anchor_positions, block_keep_mask, seq_len, block_size, dtype):
    """Dense additive bias [B, 1, q_len, KV]; 0 where allowed, MASK_FILL elsewhere.

    A draft query in block b attends to context tokens < anchor_pos[b] and to draft
    tokens within the same block; invalid (not-kept) blocks are fully masked.
    """
    B, num_blocks = anchor_positions.shape
    q_len = num_blocks * block_size
    kv_len = seq_len + q_len

    q_block_id = mx.arange(q_len) // block_size                 # [q_len]
    kv_idx = mx.arange(kv_len)                                  # [KV]
    is_context = kv_idx < seq_len
    is_draft = kv_idx >= seq_len
    kv_block_id = (kv_idx - seq_len) // block_size

    anchor_pos = anchor_positions[:, q_block_id]                # [B, q_len]
    mask_context = is_context[None, None, :] & (kv_idx[None, None, :] < anchor_pos[:, :, None])
    mask_draft = is_draft[None, None, :] & (q_block_id[None, :, None] == kv_block_id[None, None, :])
    is_valid_block = block_keep_mask[:, q_block_id]             # [B, q_len]
    allowed = (mask_context | mask_draft) & is_valid_block[:, :, None]   # [B, q_len, KV]

    bias = mx.where(allowed, mx.array(0.0, dtype), mx.array(MASK_FILL, dtype))
    return bias[:, None, :, :]                                  # [B, 1, q_len, KV]


def build_eval_mask(seq_len, loss_mask, label_indices, safe_label_indices, block_keep_mask):
    """[B, na, bs] bool: a slot is supervised only while it stays in-range, its
    label token is in loss_mask, its block is kept, AND all earlier slots were too
    (cumprod -> contiguous enabled prefix)."""
    target_valid = label_indices < seq_len
    B, na, bs = label_indices.shape
    lm_exp = mx.broadcast_to(loss_mask[:, None, :], (B, na, loss_mask.shape[1]))
    target_loss = mx.take_along_axis(lm_exp, safe_label_indices, axis=2)
    eval_mask = target_valid & (target_loss > 0.5) & block_keep_mask[:, :, None]
    return mx.cumprod(eval_mask.astype(mx.int32), axis=-1) > 0
