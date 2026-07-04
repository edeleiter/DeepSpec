"""Single-GPU Muon optimizer for the DSpark draft's 2D hidden weight matrices.

Muon ("MomentUm Orthogonalized by Newton-schulz", Keller Jordan 2024) takes an
SGD-momentum step and orthogonalizes it via a few Newton-Schulz iterations
(approximating the polar factor UV^T of the momentum matrix). It is steepest
descent under the spectral norm, and applies only to 2D hidden weights -- the
DSpark draft's per-layer attention/MLP projections and the `fc` fusion. All 1D
gains, embeddings, and the vocab-sized markov/lm heads stay on AdamW; the split
is done in BF16Optimizer.

This is the single-GPU (world_size==1) form: no distributed sharding/all-gather,
so the whole thing is ~15 lines of core math. The RMS-matched update scale
(0.2 * sqrt(max(fan_out, fan_in)), the Moonshot recipe) makes the Muon
per-element update RMS comparable to Adam's m/sqrt(v), so Muon can reuse the
AdamW learning rate and cosine schedule rather than needing a separate ~30x LR.
"""

import torch


def _zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7):
    """Orthogonalize a 2D matrix via the quintic Newton-Schulz iteration.

    Coefficients are KellerJordan's proven-stable quintic. Runs in bf16 for
    speed; the initial spectral-norm normalization bounds every iterate, so bf16
    is safe. Returns a semi-orthogonal matrix, same shape as G. An all-zero
    gradient maps to a zero update (norm -> eps, X -> 0), so no NaN.
    """
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X = X / (X.norm() + eps)              # bound the spectrum before iterating
    transposed = X.size(0) > X.size(1)    # keep X @ X.T on the smaller dim
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """Single-GPU Muon: SGD-momentum + Newton-Schulz orthogonalization.

    Moonshot RMS-matched variant: the orthogonalized update is scaled by
    0.2 * sqrt(max(fan_out, fan_in)) so its per-element RMS matches an Adam
    update, letting Muon reuse the AdamW learning rate and share its scheduler.
    Intended for 2D hidden weights only.
    """

    def __init__(self, params, lr=6e-4, momentum=0.95, nesterov=True,
                 weight_decay=0.0, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        weight_decay=weight_decay, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr, mu = group["lr"], group["momentum"]
            wd, ns = group["weight_decay"], group["ns_steps"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(mu).add_(g)
                d = g.add(buf, alpha=mu) if group["nesterov"] else buf
                o = _zeropower_via_newtonschulz5(d, steps=ns).to(g.dtype)
                o.mul_(0.2 * (max(p.size(0), p.size(1)) ** 0.5))   # RMS-match Adam
                if wd != 0:
                    p.mul_(1 - lr * wd)                            # decoupled wd
                p.add_(o, alpha=-lr)


__all__ = ["Muon"]
