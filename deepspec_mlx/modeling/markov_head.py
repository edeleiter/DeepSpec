"""Vanilla Markov head in MLX — port of deepspec/modeling/dspark/markov_head.py.

Vanilla = memoryless logit bias: base_logits + markov_w2(markov_w1[prev_token]).
Gated/RNN variants are deferred (the canary config uses 'vanilla'). Only the
training-path methods (apply_block_logits, get_prev_embeddings) are ported here;
autoregressive sampling (sample_block_tokens) lands with the eval port (M6).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class VanillaMarkov(nn.Module):
    def __init__(self, *, vocab_size: int, markov_rank: int):
        super().__init__()
        assert markov_rank > 0
        self.vocab_size = int(vocab_size)
        self.markov_rank = int(markov_rank)
        self.markov_w1 = nn.Embedding(self.vocab_size, self.markov_rank)
        self.markov_w2 = nn.Linear(self.markov_rank, self.vocab_size, bias=False)

    def get_prev_embeddings(self, token_ids: mx.array) -> mx.array:
        return self.markov_w1(token_ids)

    def compute_step_bias(self, token_ids: mx.array) -> mx.array:
        return self.markov_w2(self.get_prev_embeddings(token_ids))

    def apply_block_logits(self, base_logits: mx.array, *, token_ids: mx.array) -> mx.array:
        # base_logits [B, na, bs, vocab]; token_ids [B, na, bs]
        if base_logits.shape[2] == 0:
            return base_logits
        return base_logits + self.compute_step_bias(token_ids)


def build_markov_head(config):
    if int(config.markov_rank) <= 0:
        return None
    assert config.markov_head_type == "vanilla", \
        f"only 'vanilla' markov ported so far, got {config.markov_head_type}"
    return VanillaMarkov(vocab_size=config.vocab_size, markov_rank=config.markov_rank)
