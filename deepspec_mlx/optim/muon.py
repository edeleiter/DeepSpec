"""Muon optimizer in MLX — a faithful port of deepspec/utils/muon.py.

Muon ("MomentUm Orthogonalized by Newton-schulz", Keller Jordan 2024) takes an
SGD-momentum step and orthogonalizes it via a few Newton-Schulz iterations
(approximating the polar factor U V^T of the momentum matrix). Steepest descent
under the spectral norm; applies only to 2D hidden weights. The RMS-matched
update scale (0.2 * sqrt(max(fan_out, fan_in)), the Moonshot recipe) makes the
per-element update RMS comparable to Adam's m/sqrt(v), so Muon reuses the AdamW
learning rate and cosine schedule.

Single-device form: no distributed all-gather (world_size==1). The MuonAdam split
lives in optimizer.py.
"""

from __future__ import annotations

import mlx.core as mx
from mlx.optimizers import Optimizer


def newton_schulz5(G: mx.array, steps: int = 5, eps: float = 1e-7) -> mx.array:
    """Orthogonalize a 2D matrix via the quintic Newton-Schulz iteration.

    Direct translation of deepspec/utils/muon.py:_zeropower_via_newtonschulz5.
    Runs in bf16; the initial Frobenius-norm normalization bounds every iterate,
    so bf16 is safe and an all-zero gradient maps to a zero update (no NaN).
    Returns a semi-orthogonal matrix with the same shape as G.
    """
    assert G.ndim == 2, "Newton-Schulz orthogonalization is defined for 2D matrices"
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.astype(mx.bfloat16)
    # Frobenius norm (matches torch X.norm()). Flatten to be unambiguous about
    # ord across mx.linalg versions (vector 2-norm of the flattened matrix).
    X = X / (mx.linalg.norm(X.reshape(-1)) + eps)
    transposed = X.shape[0] > X.shape[1]     # keep X @ X.T on the smaller dim
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class Muon(Optimizer):
    """Single-device Muon: SGD-momentum + Newton-Schulz orthogonalization.

    Intended for 2D hidden weights only (routed by the MuonAdam split). Mirrors
    the torch Muon.step(): buf = mu*buf + g; d = g + mu*buf (nesterov); orthogonalize
    d; scale by 0.2*sqrt(max(fan_out, fan_in)); decoupled weight decay; p -= lr*o.
    """

    def __init__(
        self,
        learning_rate,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
    ):
        super().__init__()
        self._maybe_schedule("learning_rate", learning_rate)
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self.ns_steps = ns_steps

    def init_single(self, parameter: mx.array, state: dict):
        state["v"] = mx.zeros_like(parameter)

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict):
        mu = self.momentum
        buf = mu * state["v"] + gradient          # buf.mul_(mu).add_(g)
        state["v"] = buf
        d = gradient + mu * buf if self.nesterov else buf  # g + mu*buf
        o = newton_schulz5(d, self.ns_steps).astype(gradient.dtype)
        fan_out, fan_in = parameter.shape
        o = o * (0.2 * (max(fan_out, fan_in) ** 0.5))      # RMS-match Adam
        lr = self.learning_rate.astype(gradient.dtype)
        p = parameter
        if self.weight_decay != 0:
            p = p * (1 - lr * self.weight_decay)           # decoupled wd
        return p - lr * o


__all__ = ["Muon", "newton_schulz5"]
