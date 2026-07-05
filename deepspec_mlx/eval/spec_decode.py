"""Speculative-decoding acceptance eval in MLX — port of the DSpark eval path
(base_evaluator.generate_decoding_sample + verify_draft_tokens + draft_ops +
evaluator._propose/_update).

The DELIVERABLE: acceptance_length, the native-MLX equivalent of the paper's Table-1
metric. Draft proposes a block; target verifies it in one forward; rejection sampling
accepts the longest matching prefix; the target KV cache rewinds via mlx-lm's
trim_prompt_cache (validated in M1). The draft runs cache-free (full-attention single
block, provably equivalent — see Qwen3DSparkModel.backbone_block).
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache


# ---------- sampling helpers (port of deepspec/utils/sampling.py) ----------
def logits_to_probs(logits, temperature):
    if temperature < 1e-5:
        am = mx.argmax(logits, axis=-1)
        return (mx.arange(logits.shape[-1]) == am[..., None]).astype(mx.float32)
    return mx.softmax(logits.astype(mx.float32) / temperature, axis=-1)


def gather_token_probs(probs, token_ids):
    return mx.take_along_axis(probs, token_ids[..., None], axis=-1)[..., 0]


def sample_from_probs(probs, key):
    # probs [..., V] -> sampled ids [...]
    return mx.random.categorical(mx.log(probs + 1e-20), key=key)


def sample_residual(target_p, draft_p, key):
    residual = mx.maximum(target_p - draft_p, 0.0)
    mass = residual.sum(axis=-1, keepdims=True)
    residual = mx.where(mass <= 1e-8, target_p, residual)
    mass = residual.sum(axis=-1, keepdims=True)
    residual = residual / mx.maximum(mass, 1e-8)
    return sample_from_probs(residual, key)


# ---------- target with KV cache + per-layer hidden capture ----------
class TargetRunner:
    def __init__(self, model, target_layer_ids):
        self.model = model
        self.tli = [int(x) for x in target_layer_ids]
        self.cache = make_prompt_cache(model)

    def forward(self, input_ids):
        """Cached forward -> (logits [1,S,V], target_hidden [1,S,L*H] concat raw layers)."""
        m = self.model.model
        a = self.model.args
        h = m.embed_tokens(input_ids)
        captured = {}
        if -1 in self.tli:                # -1 = embedding output
            captured[-1] = h
        mask = create_attention_mask(h, self.cache[0])
        want = set(self.tli)
        for i, (layer, c) in enumerate(zip(m.layers, self.cache)):
            h = layer(h, mask, c)
            if i in want:
                captured[i] = h
        last = m.norm(h)
        logits = m.embed_tokens.as_linear(last) if a.tie_word_embeddings else self.model.lm_head(last)
        target_hidden = mx.concatenate([captured[i] for i in self.tli], axis=-1)  # tli order
        return logits, target_hidden

    @property
    def offset(self):
        return self.cache[0].offset

    def trim(self, n):
        if n > 0:
            trim_prompt_cache(self.cache, int(n))


# ---------- markov autoregressive block sampling ----------
def markov_sample_block(draft, base_logits, first_prev_token, temperature, key):
    """Autoregressive within-block sampling with the vanilla markov bias.
    base_logits [1, bs, V] -> sampled [1, bs], corrected_logits [1, bs, V]."""
    bs = base_logits.shape[1]
    prev = first_prev_token                       # [1]
    sampled, corrected = [], []
    keys = mx.random.split(key, bs)
    for step in range(bs):
        step_logits = base_logits[:, step, :]
        if draft.markov_head is not None:
            step_logits = step_logits + draft.markov_head.compute_step_bias(prev)
        corrected.append(step_logits)
        if temperature < 1e-5:
            tok = mx.argmax(step_logits, axis=-1)
        else:
            tok = mx.random.categorical(step_logits / temperature, key=keys[step])
        sampled.append(tok)
        prev = tok
    return mx.stack(sampled, axis=1), mx.stack(corrected, axis=1)


# ---------- confidence-head early exit (port of draft_ops._confident_prefix_length) ----------
def confident_prefix_length(draft, block_hidden, first_token, sampled, threshold, block_size):
    """Truncate the proposal at the first draft position whose confidence < threshold.
    threshold<=0 (canary default) or no confidence head -> full block (no truncation)."""
    if threshold <= 0.0 or draft.confidence_head is None:
        return block_size
    prev = mx.concatenate([first_token[:, None], sampled[:, :-1]], axis=1)   # [1, bs]
    if draft.confidence_head_with_markov:
        feats = mx.concatenate([block_hidden, draft.markov_head.get_prev_embeddings(prev)], axis=-1)
    else:
        feats = block_hidden
    conf = draft.confidence_head(feats)                        # [1, bs]
    below = [bool(x) for x in (mx.sigmoid(conf.astype(mx.float32)) < threshold)[0]]
    return below.index(True) if any(below) else block_size


# ---------- the spec-decode loop ----------
def generate(target: TargetRunner, draft, input_ids, *, max_new_tokens, block_size,
             temperature=0.0, stop_ids=None, confidence_threshold=0.0, seed=0):
    stop_ids = set(int(x) for x in (stop_ids or []))
    key = mx.random.key(seed)
    num_input = input_ids.shape[1]
    max_length = num_input + max_new_tokens

    logits, target_hidden = target.forward(input_ids)          # prefill
    key, sub = mx.random.split(key)
    first = sample_from_probs(logits_to_probs(logits[:, -1, :], temperature), sub)  # [1]
    mx.eval(first)

    committed = [int(first[0])]
    full_target_hidden = target_hidden                         # context 0..num_input-1
    start = num_input
    acceptance_lengths, proposal_lengths = [], []
    pos_accept = [0] * block_size                              # per-position accepted count
    pos_total = [0] * block_size                               # per-position proposed count
    if committed[-1] in stop_ids:
        return SimpleNamespace(committed=committed, acceptance_lengths=[], proposal_lengths=[],
                               num_input=num_input, num_output=len(committed),
                               accept_sum=0, proposal_len_sum=0, proposal_count=0,
                               pos_accept=pos_accept, pos_total=pos_total)

    mask_id = draft.mask_token_id
    while start < max_length:
        current = mx.array([committed[-1]], dtype=mx.int32)     # [1]
        draft_input_ids = mx.concatenate(
            [current[:, None], mx.full((1, block_size - 1), mask_id, dtype=mx.int32)], axis=1)

        block_hidden = draft.backbone_block(full_target_hidden, draft_input_ids)  # [1, bs, H]
        base_logits = draft.lm_head(block_hidden)
        key, ksamp = mx.random.split(key)
        sampled, corrected = markov_sample_block(draft, base_logits, current, temperature, ksamp)

        # confidence early-exit: keep only the first k confident draft tokens
        k = confident_prefix_length(draft, block_hidden, current, sampled, confidence_threshold, block_size)
        sampled = sampled[:, :k]
        draft_probs = logits_to_probs(corrected[:, :k, :], temperature) if k > 0 else None  # [1, k, V]

        verify_input_ids = mx.concatenate([current[:, None], sampled], axis=1).astype(mx.int32)  # [1, k+1]
        vlogits, vhidden = target.forward(verify_input_ids)
        target_probs = logits_to_probs(vlogits, temperature)   # [1, k+1, V]

        accept_prefix = None
        if k > 0:
            sel_t = gather_token_probs(target_probs[:, :-1, :], sampled)
            sel_d = mx.maximum(gather_token_probs(draft_probs, sampled), 1e-8)
            accept_prob = mx.minimum(sel_t / sel_d, 1.0)       # [1, k]
            key, kacc = mx.random.split(key)
            rand = mx.random.uniform(shape=accept_prob.shape, key=kacc)
            accept_prefix = [int(x) for x in mx.cumprod((rand < accept_prob).astype(mx.int32), axis=1)[0]]
            accepted = sum(accept_prefix)
        else:
            accepted = 0

        # stop token inside accepted prefix
        terminated = False
        for j, t in enumerate([int(x) for x in sampled[0, :accepted]]):
            if t in stop_ids:
                accepted = j + 1
                terminated = True
                break

        # metrics: per-position accept over the k proposed positions (pre-stop accept_prefix)
        for j in range(k):
            pos_total[j] += 1
            pos_accept[j] += accept_prefix[j]

        committed.extend(int(x) for x in sampled[0, :accepted])
        effective = accepted if terminated else k
        proposal_lengths.append(effective)                     # matches base_evaluator.py:407

        if terminated:
            acceptance_lengths.append(accepted)
            start += accepted
            target.trim((k + 1) - accepted)
            break

        # bonus/correction token
        if accepted < k:
            key, kres = mx.random.split(key)
            next_token = int(sample_residual(target_probs[0, accepted, :], draft_probs[0, accepted, :], kres))
        else:
            key, kb = mx.random.split(key)
            next_token = int(sample_from_probs(target_probs[0, -1, :], kb))
        committed.append(next_token)
        acceptance_lengths.append(accepted + 1)
        start += accepted + 1
        target.trim(k - accepted)                              # keep start = prev+accepted+1
        full_target_hidden = mx.concatenate([full_target_hidden, vhidden[:, :accepted + 1, :]], axis=1)
        mx.eval(full_target_hidden)
        if next_token in stop_ids:
            break

    return SimpleNamespace(
        committed=committed, acceptance_lengths=acceptance_lengths,
        proposal_lengths=proposal_lengths, num_input=num_input,
        num_output=len(committed),
        accept_sum=sum(acceptance_lengths), proposal_len_sum=sum(proposal_lengths),
        proposal_count=len(proposal_lengths), pos_accept=pos_accept, pos_total=pos_total,
    )
