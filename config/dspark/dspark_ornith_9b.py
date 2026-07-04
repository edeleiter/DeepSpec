import os
from deepspec.trainer import Qwen35DSparkTrainer
BASE_TB_DIR = os.path.expanduser("~/tensorboard")
BASE_CKPT_DIR = os.path.expanduser("~/checkpoints")
project_name = "deepspec"
exp_name = "dspark_ornith_9b"
seed = 42

# DSpark draft targeting Ornith-1.0-9B (Qwen3.5). This is the MVP / proof-of-
# concept config from docs/plans: single Linux CUDA GPU, tiny cache, slashed
# num_anchors so the [batch, num_anchors, block, vocab=248320] logits tensor
# fits in memory. Scale up only after the MVP gates pass.
model = dict(
    # A HF id (deepreinforce-ai/Ornith-1.0-9B) or a local path to the bf16
    # safetensors. The Q4_K_M GGUF is NOT usable here.
    target_model_name_or_path="deepreinforce-ai/Ornith-1.0-9B",
    block_size=7,
    num_draft_layers=3,
    # Ornith layer_types repeat [linear, linear, linear, full] x8, so the
    # full-attention layers are {3,7,11,15,19,23,27,31}. Those carry global
    # context and are the right hidden states to distill. Start with 3 of them;
    # confirm indices with scripts/ornith/check_load.py.
    target_layer_ids=[7, 15, 23],
    # Ornith's dedicated pad token (config.pad_token_id=248044), a neutral
    # placeholder for the masked draft positions. Confirmed valid via
    # scripts/ornith/check_load.py (eos is 248046 if you prefer that instead).
    mask_token_id=248044,
    num_anchors=16,

    ## markov head
    markov_rank=256,
    markov_head_type='vanilla',

    ## confidence head
    confidence_head_alpha=1.0,
    confidence_head_with_markov=True,

    ## loss
    loss_decay_gamma=4.0,
    ce_loss_alpha=0.1,
    l1_loss_alpha=0.9,
)

train = dict(
    trainer_cls=Qwen35DSparkTrainer,
    lr=6.0e-4,
    warmup_ratio=0.04,
    weight_decay=0.0,
    precision="bf16",
    local_batch_size=1,
    # Small global batch for a single-GPU MVP (raise as hardware allows).
    global_batch_size=64,
    num_train_epochs=10,
    max_train_steps=None,
    max_grad_norm=1.0,
    sharding_strategy="no_shard",
    # Keep compile OFF for the first runs: flex_attention + torch.compile +
    # triton are the fragile part of the stack. Turn on once training is stable.
    torch_compile=False,
)

logging = dict(
    logging_steps=10,
    checkpointing_steps=500,
)

data = dict(
    target_cache_path=None,
    chat_template="qwen",
    # Short sequences keep the target cache small (bytes/token scales with this).
    max_length=1024,
    num_workers=4,
)


def finalize_cfg(cfg):
    logging_cfg = dict(cfg["logging"])
    project_name=str(cfg['project_name'])
    exp_name = str(cfg["exp_name"])
    logging_cfg["checkpoint_dir"] = os.path.join(BASE_CKPT_DIR, project_name, exp_name)
    logging_cfg["tensorboard_dir"] = os.path.join(BASE_TB_DIR, project_name, exp_name)
    cfg["logging"] = logging_cfg

    return cfg
