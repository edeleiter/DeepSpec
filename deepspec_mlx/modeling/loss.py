"""DSpark loss in MLX — port of deepspec/modeling/dspark/loss.py at world_size==1.

At world_size==1 the all-reduce is a no-op and the *world_size multiply is *1, so
the backward loss is just:
    ce_alpha*CE + l1_alpha*L1 + conf_alpha*Conf
with per-position exponential decay weighting. CE/softmax run in fp32 (the memory
hogs); L1 reuses accept_rate to avoid recomputing softmaxes (as the torch code does).
Returns (loss, metrics) — metrics are plain floats/arrays for logging (no distributed
metric system).
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx

from .dspark_common import DSparkForwardOutput


def _loss_weight_mask(eval_mask, block_size, loss_decay_gamma):
    lwm = eval_mask.astype(mx.float32)
    if loss_decay_gamma and loss_decay_gamma > 0:
        pos = mx.arange(block_size, dtype=mx.float32)[None, None, :]
        lwm = lwm * mx.exp(-pos / float(loss_decay_gamma))
    return lwm


def _accept_rate_3d(draft_logits, aligned_target_logits):
    if aligned_target_logits is None:
        return None
    dp = mx.softmax(draft_logits.astype(mx.float32), axis=-1)
    tp = mx.softmax(aligned_target_logits.astype(mx.float32), axis=-1)
    ar = 1.0 - 0.5 * mx.abs(dp - tp).sum(axis=-1)
    return mx.clip(ar, 0.0, 1.0)


def _ce_per_token(logits_2d, targets_1d):
    # cross entropy = logsumexp(logits) - logit[target], in fp32
    lse = mx.logsumexp(logits_2d, axis=-1)
    tgt = mx.take_along_axis(logits_2d, targets_1d[:, None], axis=-1)[:, 0]
    return lse - tgt


def _bce_with_logits(z, y):
    # stable binary cross entropy with logits: max(z,0) - z*y + log(1+exp(-|z|))
    return mx.maximum(z, 0.0) - z * y + mx.log1p(mx.exp(-mx.abs(z)))


def compute_dspark_loss(
    outputs: DSparkForwardOutput,
    *,
    loss_decay_gamma: Optional[float],
    ce_loss_alpha: float,
    l1_loss_alpha: float,
    confidence_head_alpha: float,
):
    dl = outputs.draft_logits
    B, na, bs, V = dl.shape
    lwm = _loss_weight_mask(outputs.eval_mask, bs, loss_decay_gamma)   # [B,na,bs]

    # CE. The logsumexp is done in fp32 for stability — matching torch F.cross_entropy,
    # which upcasts internally. Crucially, `dl` starts in the model's compute dtype, so in
    # bf16 mode these logits are already bf16 (same as the oracle's bf16 draft_logits) BEFORE
    # the upcast -> the out-of-band CE parity (~1e-4) is achievable in bf16 mode. (fixes QA M2)
    ce_per = _ce_per_token(dl.reshape(-1, V).astype(mx.float32), outputs.target_ids.reshape(-1))
    flat_w = lwm.reshape(-1)
    ce_num = (ce_per * flat_w).sum()
    ce_den = flat_w.sum()

    ar = _accept_rate_3d(dl, outputs.aligned_target_logits)

    # L1 (total variation = 2*(1-accept_rate)), reusing accept_rate
    if l1_loss_alpha > 0 and ar is not None:
        l1_num = (2.0 * (1.0 - ar) * lwm).sum()
        l1_den = lwm.sum()
    else:
        l1_num = mx.array(0.0)
        l1_den = mx.array(0.0)

    # Confidence BCE against detached accept_rate
    has_conf = outputs.confidence_pred is not None
    if has_conf:
        assert ar is not None, "confidence head needs aligned_target_logits"
        ct = mx.stop_gradient(ar)
        conf_num = (_bce_with_logits(outputs.confidence_pred.astype(mx.float32), ct) * lwm).sum()
        conf_den = lwm.sum()
    else:
        conf_num = mx.array(0.0)
        conf_den = mx.array(0.0)

    ce = ce_num / (ce_den + 1e-6)
    l1 = l1_num / (l1_den + 1e-6)
    conf = conf_num / (conf_den + 1e-6)
    loss = ce_loss_alpha * ce + l1_loss_alpha * l1 + confidence_head_alpha * conf

    # metrics (per-position mean accept rate)
    metrics = {"ce_loss": ce, "l1_loss": l1, "confidence_loss": conf, "loss": loss}
    if ar is not None:
        em = outputs.eval_mask.astype(mx.float32)
        pos_accept = (ar * em).sum(axis=(0, 1))         # [bs]
        pos_count = em.sum(axis=(0, 1)) + 1e-6          # [bs]
        metrics["accept_rate_per_pos"] = pos_accept / pos_count
    return loss, metrics
