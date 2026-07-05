"""MLX modeling: instrumented target capture (M3) and the DSpark draft (M4)."""

from .qwen3_target_capture import capture_hidden_states, model_dims, target_embed_and_head
from .config import DSparkDraftConfig, build_draft_config
from .dspark_qwen3 import Qwen3DSparkModel
from .dspark_common import DSparkForwardOutput
from .loss import compute_dspark_loss

__all__ = [
    "capture_hidden_states",
    "model_dims",
    "target_embed_and_head",
    "DSparkDraftConfig",
    "build_draft_config",
    "Qwen3DSparkModel",
    "DSparkForwardOutput",
    "compute_dspark_loss",
]
