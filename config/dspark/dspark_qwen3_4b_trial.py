import os
from deepspec.trainer import Qwen3DSparkTrainer
BASE_TB_DIR = os.path.expanduser("~/tensorboard")
BASE_CKPT_DIR = os.path.expanduser("~/checkpoints")
project_name = "deepspec"
exp_name = "dspark_qwen3_4b_trial"
seed = 42

# Single-GPU (16 GB) trial config for a DSpark draft targeting Qwen/Qwen3-4B.
# This is a FAITHFUL copy of config/dspark/dspark_qwen3_4b.py: the quality knobs
# (target_layer_ids, block_size, num_draft_layers, max_length, loss weights) are
# left at stock values. Only the knobs that would OOM or bottleneck a single
# consumer GPU are dialed down -- see the inline notes. Scale toward the stock
# config as VRAM/time allow. Drive it with scripts/qwen3_4b/run_pipeline.sh.
model = dict(
    target_model_name_or_path="Qwen/Qwen3-4B",
    block_size=7,
    num_draft_layers=5,
    # Stock Qwen3-4B distillation layers. Qwen3-4B has 36 layers (0..35); max=33
    # excludes the final layer, satisfying assert_no_final_target_layer in eval
    # (deepspec/eval/base_evaluator.py:100-112). Keep as-is.
    target_layer_ids=[1, 9, 17, 25, 33],
    mask_token_id=151669,
    # #1 OOM knob: drives the [1, num_anchors, block_size, vocab=151936] training
    # logits tensor. Stock is 512. Start at 256 on 16 GB; raise toward 512 if no
    # OOM, drop to 128 if OOM.
    num_anchors=256,

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
    trainer_cls=Qwen3DSparkTrainer,
    lr=6.0e-4,
    warmup_ratio=0.04,
    weight_decay=0.0,
    precision="bf16",
    local_batch_size=1,
    # Stock is 512. Smaller global batch -> more optimizer steps per epoch and
    # less grad-accum latency on a single GPU. Grad accumulation still applies
    # (global_batch_size / (local_batch_size * num_gpus) micro-steps per step).
    global_batch_size=64,
    num_train_epochs=10,
    max_train_steps=None,
    max_grad_norm=1.0,
    sharding_strategy="no_shard",
    # Keep compile OFF for the first runs: flex_attention + torch.compile + triton
    # is the fragile part of the stack. Turn on once training is stable.
    torch_compile=False,
)

logging = dict(
    logging_steps=10,
    # A final checkpoint is always written at end-of-training (base_trainer.py:400),
    # so eval has step_latest even on short smoke runs. This interval only governs
    # intermediate checkpoints (each of which also auto-launches eval).
    checkpointing_steps=500,
)

data = dict(
    target_cache_path=None,
    chat_template="qwen",
    max_length=4096,
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
