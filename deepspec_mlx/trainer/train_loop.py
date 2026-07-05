"""Single-device DSpark training loop in MLX — port of the essentials of
deepspec/trainer/{base_trainer,dspark_trainer}.py.

All the distributed machinery (FSDP, NCCL, CUDAPrefetcher, mp.spawn) is deleted —
one device, unified memory. Keeps: value_and_grad, manual gradient accumulation with
mx.eval per micro-step (bounds the lazy graph), grad clipping, the Muon+AdamW split
optimizer (deepspec_mlx/optim), and the cosine-warmup schedule.

Precision (scheme C): the optimizer keeps an fp32 MASTER of the trainable params;
the forward/backward run in the model's `compute_dtype` (fp32 by default for the
canary; set bf16 for oracle parity + scaling). Grads are cast to fp32, the master is
updated, then cast back into the model. When compute_dtype==fp32 the casts are no-ops,
so this degenerates to plain fp32. embed_tokens and lm_head are frozen (copied from
the target). `assert_uniform_dtype` guards against any accidental mixed-precision model.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.optimizers import clip_grad_norm
from mlx.utils import tree_map

from deepspec_mlx.modeling.loss import compute_dspark_loss
from deepspec_mlx.optim import build_muon_adam, cosine_warmup


def _batch_dims(sample):
    return (
        sample["input_ids"][None],
        sample["target_hidden_states"][None],
        sample["loss_mask"][None].astype(mx.float32),
        sample["target_last_hidden_states"][None],
    )


def freeze_frozen_heads(draft):
    """Freeze the target-copied embed_tokens + lm_head so they're excluded from grads."""
    draft.embed_tokens.freeze()
    draft.lm_head.freeze()


def accept_rate_report(draft, samples, loss_cfg, key):
    """No-grad pass: mean per-position accept_rate + mean CE over samples."""
    rates, ces = [], []
    for i, s in enumerate(samples):
        ii, ths, lm, tlhs = _batch_dims(s)
        out = draft(ii, ths, lm, tlhs, key=mx.random.split(key, len(samples))[i])
        _, m = compute_dspark_loss(out, **loss_cfg)
        rates.append(m["accept_rate_per_pos"])
        ces.append(float(m["ce_loss"]))
    ar = np.array(mx.mean(mx.stack(rates), axis=0))
    return ar, float(np.mean(ces))


# --- precision scheme C: fp32 master, compute-dtype forward (see plan 1.1) ---
def make_master(draft):
    """fp32 master copy of the trainable params (the optimizer operates on this)."""
    return tree_map(lambda a: a.astype(mx.float32), draft.trainable_parameters())


def load_master(draft, master):
    """Cast the fp32 master into the model at its compute dtype (no-op in fp32 mode)."""
    draft.update(tree_map(lambda a: a.astype(draft.compute_dtype), master))


def train_step(draft, opt, master, samples, loss_cfg, key, max_grad_norm=1.0):
    """One optimizer step over `samples` (grad accumulation = len(samples)).

    Forward/backward run in the model's compute dtype; grads are cast to fp32 and
    the optimizer updates the fp32 master, which is then cast back into the model.
    Returns the updated master. When compute_dtype==fp32 the casts are no-ops.
    """
    keys = mx.random.split(key, len(samples))
    acc = None
    total = 0.0
    for i, s in enumerate(samples):
        ii, ths, lm, tlhs = _batch_dims(s)

        def loss_fn(model, ii=ii, ths=ths, lm=lm, tlhs=tlhs, k=keys[i]):
            out = model(ii, ths, lm, tlhs, key=k)
            loss, _ = compute_dspark_loss(out, **loss_cfg)
            return loss

        loss, grads = nn.value_and_grad(draft, loss_fn)(draft)
        grads = tree_map(lambda g: g.astype(mx.float32), grads)   # accumulate in fp32
        acc = grads if acc is None else tree_map(lambda a, b: a + b, acc, grads)
        mx.eval(acc)                                  # bound the lazy graph per micro-step
        total += float(loss)
    acc = tree_map(lambda g: g / len(samples), acc)
    acc, gnorm = clip_grad_norm(acc, max_grad_norm)
    master = opt.apply_gradients(acc, master)         # fp32 master update
    load_master(draft, master)
    mx.eval(master, draft.parameters(), opt.state)
    return master, total / len(samples), float(gnorm)


def overfit(draft, samples, *, steps=100, lr=6e-4, warmup_ratio=0.04, seed=0,
            loss_cfg=None, log_every=10):
    """Overfit `samples` and watch CE fall / accept_rate rise off the untrained floor."""
    if loss_cfg is None:
        loss_cfg = dict(loss_decay_gamma=4.0, ce_loss_alpha=0.1,
                        l1_loss_alpha=0.9, confidence_head_alpha=1.0)
    draft.assert_uniform_dtype()                      # guardrail: no hybrid precision
    freeze_frozen_heads(draft)
    schedule = cosine_warmup(lr, steps, warmup_ratio)
    opt = build_muon_adam(schedule, use_muon=True)
    master = make_master(draft)
    opt.init(master)

    key = mx.random.key(seed)
    ar0, ce0 = accept_rate_report(draft, samples, loss_cfg, mx.random.key(1234))
    print(f"  step   0 (init): ce={ce0:.3f}  accept_rate(mean)={ar0.mean():.4f}  pos0={ar0[0]:.4f}")
    history = [(0, ce0, float(ar0.mean()))]
    for step in range(1, steps + 1):
        key, sub = mx.random.split(key)
        master, mean_loss, gnorm = train_step(draft, opt, master, samples, loss_cfg, sub)
        if step % log_every == 0 or step == steps:
            ar, ce = accept_rate_report(draft, samples, loss_cfg, mx.random.key(1234))
            print(f"  step {step:3d}: loss={mean_loss:.3f} ce={ce:.3f} "
                  f"accept_rate(mean)={ar.mean():.4f} pos0={ar[0]:.4f} gnorm={gnorm:.2f}")
            history.append((step, ce, float(ar.mean())))
    return history
