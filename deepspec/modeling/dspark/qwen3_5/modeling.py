"""DSpark draft model for Qwen3.5-family targets (e.g. Ornith-1.0-9B).

The draft architecture is identical to the Qwen3 draft: a small full-attention
transformer that consumes the target's cached hidden states as attention
context. Only the target-config extraction differs (see ``config.py``), so the
model here is a thin alias over ``Qwen3DSparkModel``. Keeping a distinct class
name matches the draft ``architectures`` field and the ``Qwen35DSparkTrainer``.
"""

from deepspec.modeling.dspark.qwen3.modeling import Qwen3DSparkModel


class Qwen35DSparkModel(Qwen3DSparkModel):
    """Qwen3.5 DSpark draft (plain full-attention; see module docstring)."""


__all__ = [
    "Qwen35DSparkModel",
]
