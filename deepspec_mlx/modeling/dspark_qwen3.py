"""Qwen3 DSpark draft model in MLX — port of deepspec/modeling/dspark/qwen3/modeling.py.

The core novelty is the custom NON-causal cross/block attention: queries come from
the noise/draft positions, keys/values are cat([target_context, noise]), masked by a
dense additive bias, with RoPE applied asymmetrically (q uses the last q_len position
slots, k uses all). Training forward only (eval sampling lands in M6).
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map

from .config import DSparkDraftConfig
from .dspark_common import (
    DSparkForwardOutput,
    apply_rope_k,
    apply_rope_q,
    build_eval_mask,
    create_dspark_attention_bias,
    create_noise_ids,
    create_position_ids,
    rope_cos_sin,
    sample_anchor_positions,
)
from .markov_head import build_markov_head


class DSparkAttention(nn.Module):
    def __init__(self, config: DSparkDraftConfig):
        super().__init__()
        d = config.head_dim
        self.head_dim = d
        self.n_heads = config.num_attention_heads
        self.n_kv = config.num_key_value_heads
        self.scaling = d ** -0.5
        b = config.attention_bias
        self.q_proj = nn.Linear(config.hidden_size, self.n_heads * d, bias=b)
        self.k_proj = nn.Linear(config.hidden_size, self.n_kv * d, bias=b)
        self.v_proj = nn.Linear(config.hidden_size, self.n_kv * d, bias=b)
        self.o_proj = nn.Linear(self.n_heads * d, config.hidden_size, bias=b)
        self.q_norm = nn.RMSNorm(d, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(d, eps=config.rms_norm_eps)

    def __call__(self, hidden, target_hidden, cos, sin, bias):
        B, q_len, _ = hidden.shape
        ctx_len = target_hidden.shape[1]
        d = self.head_dim
        q = self.q_proj(hidden).reshape(B, q_len, self.n_heads, d)
        q = self.q_norm(q).transpose(0, 2, 1, 3)                       # [B,Hq,q_len,d]
        k = mx.concatenate([self.k_proj(target_hidden), self.k_proj(hidden)], axis=1)
        v = mx.concatenate([self.v_proj(target_hidden), self.v_proj(hidden)], axis=1)
        k = self.k_norm(k.reshape(B, ctx_len + q_len, self.n_kv, d)).transpose(0, 2, 1, 3)
        v = v.reshape(B, ctx_len + q_len, self.n_kv, d).transpose(0, 2, 1, 3)  # [B,Hkv,KV,d]
        q = apply_rope_q(q, cos, sin)
        k = apply_rope_k(k, cos, sin)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scaling, mask=bias)
        out = out.transpose(0, 2, 1, 3).reshape(B, q_len, self.n_heads * d)
        return self.o_proj(out)


class DSparkMLP(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class DSparkDecoderLayer(nn.Module):
    def __init__(self, config: DSparkDraftConfig):
        super().__init__()
        self.self_attn = DSparkAttention(config)
        self.mlp = DSparkMLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, target_hidden, hidden, cos, sin, bias):
        hidden = hidden + self.self_attn(self.input_layernorm(hidden), target_hidden, cos, sin, bias)
        hidden = hidden + self.mlp(self.post_attention_layernorm(hidden))
        return hidden


class AcceptRatePredictor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = nn.Linear(int(input_dim), 1)

    def __call__(self, x):
        return self.proj(x)[..., 0]


class Qwen3DSparkModel(nn.Module):
    def __init__(self, config: DSparkDraftConfig, compute_dtype=mx.float32):
        super().__init__()
        self.config = config
        self.block_size = config.block_size
        self.mask_token_id = config.mask_token_id
        self.num_anchors = config.num_anchors
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta
        self.compute_dtype = compute_dtype

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [DSparkDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.fc = nn.Linear(len(config.target_layer_ids) * config.hidden_size, config.hidden_size, bias=False)
        self.hidden_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.markov_head = build_markov_head(config)
        self.confidence_head_with_markov = bool(config.confidence_head_with_markov)
        self.confidence_head = None
        if config.enable_confidence_head:
            input_dim = config.hidden_size + (config.markov_rank if self.confidence_head_with_markov else 0)
            self.confidence_head = AcceptRatePredictor(input_dim)

        # mlx.nn builds fp32 params; cast the whole model to the compute dtype so
        # there is ONE well-defined precision (no accidental hybrid). fp32 is a no-op.
        if compute_dtype != mx.float32:
            self.set_compute_dtype(compute_dtype)

    # --- precision (scheme C: one compute dtype; fp32 master lives in the trainer) ---
    def set_compute_dtype(self, dtype):
        """Cast every parameter to `dtype`. The trainer keeps the fp32 master."""
        self.compute_dtype = dtype
        self.update(tree_map(lambda a: a.astype(dtype), self.parameters()))

    def assert_uniform_dtype(self):
        """Guardrail: every param must share compute_dtype (kills the hybrid bug)."""
        bad = [(k, v.dtype) for k, v in tree_flatten(self.parameters())
               if v.dtype != self.compute_dtype]
        assert not bad, f"non-uniform param dtypes (expected {self.compute_dtype}): {bad[:5]}"

    # --- weight init from the (frozen) target ---
    def initialize_from_target(self, embed_weight: mx.array, lm_head_weight: mx.array):
        """Copy the target's embed_tokens and lm_head into the (frozen) draft heads.

        Takes the head SEPARATELY (no shared-array aliasing) so it is correct for
        both tied (Qwen3-0.6B/1.7B: pass embed as the head) and untied (4B+) targets.
        Both are cast to the model's compute dtype (never left at the target's bf16).
        """
        assert self.embed_tokens.weight.shape == embed_weight.shape
        assert self.lm_head.weight.shape == lm_head_weight.shape
        self.embed_tokens.weight = embed_weight.astype(self.compute_dtype)
        self.lm_head.weight = lm_head_weight.astype(self.compute_dtype)
        self.assert_uniform_dtype()

    def _param_dtype(self):
        return self.compute_dtype

    def backbone_block(self, full_target_hidden: mx.array, draft_input_ids: mx.array) -> mx.array:
        """Cache-free single-block draft forward for eval (M6).

        The draft block at anchor=start attends to ALL context (positions < start)
        and its own block, with no masking -> plain full attention (bias=None).
        Recomputes over the full context each call (O(n^2), fine for the canary).
        Returns output_hidden [B, block_size, H].
        """
        from .dspark_common import rope_cos_sin
        dt = self._param_dtype()
        fth = full_target_hidden.astype(dt)
        noise = self.embed_tokens(draft_input_ids)
        target_hidden = self.hidden_norm(self.fc(fth))         # [B, ctx_len, H]
        ctx_len = target_hidden.shape[1]
        bs = draft_input_ids.shape[1]
        full_pos = mx.arange(ctx_len + bs, dtype=mx.int32)[None, :]
        cos, sin = rope_cos_sin(full_pos, self.head_dim, self.rope_theta, dtype=dt)
        hidden = noise
        for layer in self.layers:
            hidden = layer(target_hidden, hidden, cos, sin, None)
        return self.norm(hidden)

    def __call__(
        self,
        input_ids: mx.array,
        target_hidden_states: mx.array,
        loss_mask: mx.array,
        target_last_hidden_states: Optional[mx.array] = None,
        *,
        anchor_positions: Optional[mx.array] = None,
        block_keep_mask: Optional[mx.array] = None,
        key=None,
    ) -> DSparkForwardOutput:
        B, S = input_ids.shape
        bs = self.block_size
        dt = self._param_dtype()
        target_hidden_states = target_hidden_states.astype(dt)
        if target_last_hidden_states is not None:
            target_last_hidden_states = target_last_hidden_states.astype(dt)

        if anchor_positions is None:
            anchor_positions, block_keep_mask = sample_anchor_positions(S, loss_mask, self.num_anchors, key=key)
        num_blocks = anchor_positions.shape[1]

        noise_ids = create_noise_ids(
            input_ids, anchor_positions, block_keep_mask,
            mask_token_id=self.mask_token_id, block_size=bs,
        )
        noise_embedding = self.embed_tokens(noise_ids)                 # [B, na*bs, H]

        context_pos = mx.broadcast_to(mx.arange(S, dtype=mx.int32)[None, :], (B, S))
        draft_pos = create_position_ids(anchor_positions, bs)
        full_pos = mx.concatenate([context_pos, draft_pos], axis=1)    # [B, S + na*bs]
        bias = create_dspark_attention_bias(anchor_positions, block_keep_mask, S, bs, dtype=dt)

        # backbone
        target_hidden = self.hidden_norm(self.fc(target_hidden_states))  # [B, S, H]
        cos, sin = rope_cos_sin(full_pos, self.head_dim, self.rope_theta, dtype=dt)
        hidden = noise_embedding
        for layer in self.layers:
            hidden = layer(target_hidden, hidden, cos, sin, bias)
        output_hidden = self.norm(hidden)                              # [B, na*bs, H]
        output_4d = output_hidden.reshape(B, num_blocks, bs, -1)

        # labels / gathers
        label_offsets = mx.arange(1, bs + 1, dtype=mx.int32)[None, None, :]
        label_indices = anchor_positions[:, :, None] + label_offsets   # [B, na, bs]
        safe = mx.minimum(label_indices, S - 1)
        safe = mx.where(block_keep_mask[:, :, None], safe, mx.zeros_like(safe))
        ii_exp = mx.broadcast_to(input_ids[:, None, :], (B, num_blocks, S))
        target_ids = mx.take_along_axis(ii_exp, safe, axis=2)          # [B, na, bs]

        aligned_target_logits = None
        if target_last_hidden_states is not None:
            H = target_last_hidden_states.shape[-1]
            tpi = mx.maximum(safe - 1, 0)                              # [B, na, bs]
            tlh_exp = mx.broadcast_to(target_last_hidden_states[:, None, :, :], (B, num_blocks, S, H))
            idx = mx.broadcast_to(tpi[:, :, :, None], (B, num_blocks, bs, H))
            aligned_hidden = mx.take_along_axis(tlh_exp, idx, axis=2)  # [B, na, bs, H]
            aligned_target_logits = self.lm_head(aligned_hidden)       # [B, na, bs, V]

        eval_mask = build_eval_mask(S, loss_mask, label_indices, safe, block_keep_mask)

        anchor_token_ids = mx.take_along_axis(input_ids, anchor_positions, axis=1)  # [B, na]
        prev_token_ids = mx.concatenate([anchor_token_ids[:, :, None], target_ids[:, :, :-1]], axis=-1)

        draft_logits = self.lm_head(output_hidden).reshape(B, num_blocks, bs, -1)
        if self.markov_head is not None:
            draft_logits = self.markov_head.apply_block_logits(draft_logits, token_ids=prev_token_ids)

        confidence_pred = None
        if self.confidence_head is not None:
            if self.confidence_head_with_markov:
                prev_emb = self.markov_head.get_prev_embeddings(prev_token_ids).astype(output_4d.dtype)
                feats = mx.concatenate([output_4d, prev_emb], axis=-1)
                confidence_pred = self.confidence_head(feats)
            else:
                confidence_pred = self.confidence_head(output_4d)

        return DSparkForwardOutput(
            draft_logits=draft_logits,
            target_ids=target_ids,
            eval_mask=eval_mask,
            block_keep_mask=block_keep_mask,
            confidence_pred=confidence_pred,
            aligned_target_logits=aligned_target_logits,
        )
