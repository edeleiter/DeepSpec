"""MuonAdam split + cosine-warmup schedule for the DSpark MLX trainer.

Mirrors deepspec/utils/optim.py:BF16Optimizer's split and schedule, but built on
MLX's MultiOptimizer (which routes params to sub-optimizers by a path predicate)
and MLX schedules. No fp32-master bookkeeping here — that lives in the training
loop (M5), which keeps params/optimizer state in fp32 and runs the forward in bf16.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.optimizers as optim

from .muon import Muon

# The torch split (optim.py:113): Muon gets 2D matrices whose name is NOT one of
# these; everything else (1D norm gains, markov head, vocab-sized heads,
# confidence head, embeddings/lm_head) stays on AdamW. MLX param paths are dotted
# (e.g. "layers.0.self_attn.q_proj.weight"), so the substring predicate carries over.
MUON_EXCLUDE = ("markov", "lm_head", "embed_tokens", "confidence_head")


def is_muon_param(path: str, weight: mx.array) -> bool:
    """MultiOptimizer filter: True -> Muon, False -> AdamW fallback."""
    return weight.ndim == 2 and not any(key in path for key in MUON_EXCLUDE)


def cosine_warmup(peak_lr: float, total_steps: int, warmup_ratio: float = 0.04):
    """Linear warmup 0 -> peak_lr over warmup_steps, then cosine decay to 0.

    Matches deepspec/utils/optim.py:CosineAnnealingWarmupLR (warmup then
    CosineAnnealingLR over the remaining steps, eta_min=0).
    """
    warmup_steps = max(1, int(warmup_ratio * total_steps))
    decay_steps = max(1, total_steps - warmup_steps)
    warmup = optim.linear_schedule(0.0, peak_lr, warmup_steps)
    cosine = optim.cosine_decay(peak_lr, decay_steps)
    return optim.join_schedules([warmup, cosine], [warmup_steps])


def build_muon_adam(
    lr_schedule,
    weight_decay: float = 0.0,
    use_muon: bool = True,
    muon_wd: float = 0.0,
):
    """Build the optimizer.

    use_muon=True  -> MultiOptimizer([Muon(matrices), AdamW(rest)]).
    use_muon=False -> plain AdamW over everything (matches DSPARK_MUON unset).

    `lr_schedule` is a peak LR float or an MLX schedule callable; Muon and AdamW
    share it (scale 1.0), stepping in lockstep exactly like the torch reference.
    A non-unity Muon LR scale would need per-optimizer schedules (deferred).
    """
    if not use_muon:
        return optim.AdamW(learning_rate=lr_schedule, weight_decay=weight_decay)

    muon = Muon(learning_rate=lr_schedule, weight_decay=muon_wd)
    adamw = optim.AdamW(learning_rate=lr_schedule, weight_decay=weight_decay)
    return optim.MultiOptimizer([muon, adamw], filters=[is_muon_param])


__all__ = ["build_muon_adam", "cosine_warmup", "is_muon_param", "MUON_EXCLUDE"]
