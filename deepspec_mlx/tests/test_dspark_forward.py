"""M4 tests for the DSpark draft forward + loss (torch-free, self-consistency + numpy).

No torch on the Mac, so instead of torch-fixture parity we validate:
  1. forward output shapes + finiteness (injected fixed anchors),
  2. the dense attention bias masks EXACTLY the intended positions,
  3. noise-id construction places the anchor token at each block start,
  4. the loss math matches an independent numpy re-implementation (CE/L1/BCE),
  5. gradients flow (finite, non-zero) through value_and_grad.

Bit-parity vs the torch reference is a later out-of-band check.

Run:  python deepspec_mlx/tests/test_dspark_forward.py
"""

from __future__ import annotations

import sys

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, __file__.rsplit("/deepspec_mlx/", 1)[0])
from deepspec_mlx.modeling.config import DSparkDraftConfig
from deepspec_mlx.modeling.dspark_qwen3 import Qwen3DSparkModel
from deepspec_mlx.modeling.dspark_common import (
    create_dspark_attention_bias,
    create_noise_ids,
    MASK_FILL,
)
from deepspec_mlx.modeling.loss import compute_dspark_loss

LG = dict(loss_decay_gamma=4.0, ce_loss_alpha=0.1, l1_loss_alpha=0.9, confidence_head_alpha=1.0)


def tiny_config():
    return DSparkDraftConfig(
        hidden_size=32, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        intermediate_size=64, vocab_size=50, rms_norm_eps=1e-6, rope_theta=1e6,
        max_position_embeddings=128, num_hidden_layers=2, num_target_layers=5,
        target_layer_ids=[1, 2, 3], block_size=3, mask_token_id=7, num_anchors=4,
        markov_rank=8, markov_head_type="vanilla",
        enable_confidence_head=True, confidence_head_with_markov=True,
    )


def make_inputs(cfg, S=10, seed=0):
    rng = np.random.default_rng(seed)
    B = 1
    L = len(cfg.target_layer_ids)
    input_ids = mx.array(rng.integers(0, cfg.vocab_size, size=(B, S)).astype(np.int32))
    loss_mask = mx.ones((B, S), dtype=mx.uint8)
    ths = mx.array(rng.standard_normal((B, S, L * cfg.hidden_size)).astype(np.float32))
    tlhs = mx.array(rng.standard_normal((B, S, cfg.hidden_size)).astype(np.float32))
    anchors = mx.array(np.array([[1, 3, 5, 7]], dtype=np.int32))
    keep = mx.array(np.array([[True, True, True, False]]))  # last block invalid
    return input_ids, loss_mask, ths, tlhs, anchors, keep


def test_forward_shapes_finite():
    cfg = tiny_config()
    m = Qwen3DSparkModel(cfg)
    input_ids, loss_mask, ths, tlhs, anchors, keep = make_inputs(cfg)
    out = m(input_ids, ths, loss_mask, tlhs, anchor_positions=anchors, block_keep_mask=keep)
    na, bs, V = cfg.num_anchors, cfg.block_size, cfg.vocab_size
    assert tuple(out.draft_logits.shape) == (1, na, bs, V), out.draft_logits.shape
    assert tuple(out.target_ids.shape) == (1, na, bs)
    assert tuple(out.eval_mask.shape) == (1, na, bs)
    assert tuple(out.confidence_pred.shape) == (1, na, bs)
    assert tuple(out.aligned_target_logits.shape) == (1, na, bs, V)
    for name, t in [("draft_logits", out.draft_logits), ("aligned", out.aligned_target_logits),
                    ("confidence", out.confidence_pred)]:
        assert bool(mx.all(mx.isfinite(t.astype(mx.float32))).item()), f"{name} not finite"
    # invalid block (index 3) must be fully masked out of eval_mask
    em = np.array(out.eval_mask.astype(mx.int32))
    assert em[0, 3].sum() == 0, "invalid block should have empty eval_mask"
    print(f"  shapes OK; eval_mask valid-block sums = {em[0].sum(axis=1).tolist()}")
    return True


def test_attention_bias_semantics():
    # anchors [2, 5], block_size 2, seq_len 6, both blocks valid.
    anchors = mx.array(np.array([[2, 5]], dtype=np.int32))
    keep = mx.array(np.array([[True, True]]))
    S, bs = 6, 2
    bias = create_dspark_attention_bias(anchors, keep, S, bs, dtype=mx.float32)
    b = np.array(bias[0, 0])  # [q_len, KV]
    q_len, KV = b.shape
    assert (q_len, KV) == (4, 6 + 4)
    allowed = b > (MASK_FILL / 2)  # True where attending allowed (0), False where masked
    # block 0 queries (q=0,1, anchor=2): context kv<2 allowed; draft kv in [6,8) allowed
    for q in (0, 1):
        assert allowed[q, 0] and allowed[q, 1] and not allowed[q, 2], "ctx< anchor only"
        assert allowed[q, 6] and allowed[q, 7], "own draft block"
        assert not allowed[q, 8] and not allowed[q, 9], "other draft block masked"
    # block 1 queries (q=2,3, anchor=5): context kv<5 allowed
    for q in (2, 3):
        assert allowed[q, 0] and allowed[q, 4] and not allowed[q, 5], "ctx< anchor"
        assert allowed[q, 8] and allowed[q, 9] and not allowed[q, 6], "own draft block"
    print("  attention bias masks context<anchor and same-block draft only")
    return True


def test_noise_ids():
    input_ids = mx.array(np.arange(10).reshape(1, 10).astype(np.int32))
    anchors = mx.array(np.array([[2, 5, 8]], dtype=np.int32))
    keep = mx.array(np.array([[True, True, False]]))
    noise = np.array(create_noise_ids(input_ids, anchors, keep, mask_token_id=99, block_size=3))
    noise = noise.reshape(3, 3)
    assert noise[0, 0] == 2 and noise[1, 0] == 5, "kept blocks -> anchor token at slot 0"
    assert noise[2, 0] == 99, "unkept block -> mask token at slot 0"
    assert (noise[:, 1:] == 99).all(), "non-first slots are mask token"
    print(f"  noise block starts = {noise[:,0].tolist()} (anchors 2,5 then mask 99)")
    return True


# ---- numpy reference for the loss ----
def np_softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def loss_numpy(out, gamma, a_ce, a_l1, a_conf):
    dl = np.array(out.draft_logits.astype(mx.float32))
    atl = np.array(out.aligned_target_logits.astype(mx.float32))
    tids = np.array(out.target_ids).astype(np.int64)
    em = np.array(out.eval_mask.astype(mx.int32)).astype(np.float32)
    cp = np.array(out.confidence_pred.astype(mx.float32))
    B, na, bs, V = dl.shape
    pos = np.arange(bs)[None, None, :]
    lwm = em * np.exp(-pos / gamma)
    # CE
    logp = dl - (dl.max(-1, keepdims=True) + np.log(np.exp(dl - dl.max(-1, keepdims=True)).sum(-1, keepdims=True)))
    ce_per = -np.take_along_axis(logp, tids[..., None], axis=-1)[..., 0]
    ce = (ce_per * lwm).sum() / (lwm.sum() + 1e-6)
    # accept rate + L1
    dp, tp = np_softmax(dl), np_softmax(atl)
    ar = np.clip(1.0 - 0.5 * np.abs(dp - tp).sum(-1), 0, 1)
    l1 = (2.0 * (1.0 - ar) * lwm).sum() / (lwm.sum() + 1e-6)
    # confidence BCE
    z = cp
    bce = np.maximum(z, 0) - z * ar + np.log1p(np.exp(-np.abs(z)))
    conf = (bce * lwm).sum() / (lwm.sum() + 1e-6)
    return a_ce * ce + a_l1 * l1 + a_conf * conf, ce, l1, conf


def test_loss_numpy_parity():
    cfg = tiny_config()
    m = Qwen3DSparkModel(cfg)
    input_ids, loss_mask, ths, tlhs, anchors, keep = make_inputs(cfg, seed=3)
    out = m(input_ids, ths, loss_mask, tlhs, anchor_positions=anchors, block_keep_mask=keep)
    loss, metrics = compute_dspark_loss(out, **LG)
    ref_loss, ref_ce, ref_l1, ref_conf = loss_numpy(out, 4.0, 0.1, 0.9, 1.0)
    dl = abs(float(loss) - float(ref_loss))
    dce = abs(float(metrics["ce_loss"]) - ref_ce)
    print(f"  loss mlx={float(loss):.6f} np={ref_loss:.6f} |Δ|={dl:.2e}; "
          f"ce Δ={dce:.2e} l1 Δ={abs(float(metrics['l1_loss'])-ref_l1):.2e} "
          f"conf Δ={abs(float(metrics['confidence_loss'])-ref_conf):.2e}")
    assert dl < 1e-4 and dce < 1e-4, "loss diverges from numpy reference"
    return True


def test_grad_flows():
    cfg = tiny_config()
    m = Qwen3DSparkModel(cfg)
    input_ids, loss_mask, ths, tlhs, anchors, keep = make_inputs(cfg, seed=5)

    def loss_fn(model):
        out = model(input_ids, ths, loss_mask, tlhs, anchor_positions=anchors, block_keep_mask=keep)
        loss, _ = compute_dspark_loss(out, **LG)
        return loss

    loss, grads = nn.value_and_grad(m, loss_fn)(m)
    from mlx.utils import tree_flatten
    flat = tree_flatten(grads)
    n_finite = sum(1 for _, g in flat if bool(mx.all(mx.isfinite(g)).item()))
    n_nonzero = sum(1 for _, g in flat if float(mx.abs(g).sum()) > 0)
    print(f"  loss={float(loss):.4f}; {n_finite}/{len(flat)} grad leaves finite, "
          f"{n_nonzero} non-zero")
    assert n_finite == len(flat), "some gradients are non-finite"
    assert n_nonzero > len(flat) // 2, "too many zero gradients"
    return True


def main():
    tests = [
        ("forward shapes + finiteness", test_forward_shapes_finite),
        ("attention bias semantics", test_attention_bias_semantics),
        ("noise-id construction", test_noise_ids),
        ("loss vs numpy reference", test_loss_numpy_parity),
        ("gradients flow", test_grad_flows),
    ]
    failed = 0
    for name, fn in tests:
        print(f"\n== {name} ==")
        try:
            fn()
            print("  PASS")
        except AssertionError as e:
            print(f"  FAIL: {e}")
            failed += 1
    print(f"\nRESULT: {'PASS — DSpark forward + loss validated' if failed == 0 else f'FAIL ({failed})'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
