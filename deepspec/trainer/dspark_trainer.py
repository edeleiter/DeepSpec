from deepspec.data import CacheCollator
from deepspec.modeling.dspark.gemma4 import Gemma4DSparkModel
from deepspec.modeling.dspark.gemma4.config import (
    build_draft_config as build_gemma4_draft_config,
)
from deepspec.modeling.dspark.loss import compute_dspark_loss
from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel
from deepspec.modeling.dspark.qwen3.config import (
    build_draft_config as build_qwen3_draft_config,
)
from deepspec.modeling.dspark.qwen3_5 import Qwen35DSparkModel
from deepspec.modeling.dspark.qwen3_5.config import (
    build_draft_config as build_qwen35_draft_config,
)
from torch.nn.attention import sdpa_kernel, SDPBackend

from deepspec.trainer.base_trainer import BaseTrainer

# sm_120 (Blackwell) at the draft's head_dim=256: the mem-efficient SDPA kernel
# runs the dense-bias attention in ~0.7 GB, while PyTorch's auto-dispatch silently
# falls to the MATH backend, which materializes full score tensors and pushes peak
# memory to 16-19 GB -> the WSL2 driver spills to system RAM over PCIe -> ~60s per
# micro-batch. Pin mem-efficient (MATH kept only as a never-expected fallback).
_DSPARK_SDPA_BACKENDS = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


class Qwen3DSparkTrainer(BaseTrainer):
    data_collator_cls = CacheCollator

    def _build_draft_model(self, *, target_config, model_args):
        import os
        draft_config = build_qwen3_draft_config(
            target_config=target_config,
            model_args=model_args,
        )
        model = Qwen3DSparkModel(draft_config)
        # On a 16 GB card the ~5.6 GB backward-activation transient pushes peak VRAM
        # to the ceiling and the WSL2 driver spills to system RAM (~60s/micro-batch).
        # Recompute activations in backward instead of storing them. Toggle off with
        # DSPARK_NO_GRAD_CKPT=1.
        if not os.environ.get("DSPARK_NO_GRAD_CKPT"):
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        return model

    # Training step.
    def run_batch(self, batch):
        import os
        # On a 16 GB card the backward-activation peak scales with sequence length
        # and spills to host RAM past ~600 tokens. Truncating to the first N tokens
        # bounds the peak; the cached target hidden states for those positions were
        # computed with full context, so they stay valid. Set DSPARK_MAX_LEN to enable.
        _maxlen = os.environ.get("DSPARK_MAX_LEN")
        if _maxlen:
            n = int(_maxlen)
            for _k in ("input_ids", "attention_mask", "loss_mask",
                       "target_hidden_states", "target_last_hidden_states"):
                if _k in batch and batch[_k].shape[1] > n:
                    batch[_k] = batch[_k][:, :n].contiguous()
        _dbg = bool(os.environ.get("DSPARK_DEBUG"))
        if _dbg:
            import torch
            g = lambda: torch.cuda.memory_allocated() / 1e9
            seq = tuple(batch["input_ids"].shape)
            prev_peak = torch.cuda.max_memory_allocated() / 1e9
            base = g()
            print(f"[dbg] seq={seq} prev_step_peak={prev_peak:.2f}GB base={base:.2f}GB", flush=True)
            torch.cuda.reset_peak_memory_stats()
        with sdpa_kernel(_DSPARK_SDPA_BACKENDS):
            outputs = self.model(
                input_ids=batch["input_ids"],
                target_hidden_states=batch["target_hidden_states"],
                loss_mask=batch["loss_mask"],
                target_last_hidden_states=batch["target_last_hidden_states"],
            )
        if _dbg:
            print(f"[dbg]   after_fwd={g():.2f}GB (fwd_peak={torch.cuda.max_memory_allocated()/1e9:.2f}) "
                  f"draft_logits={tuple(outputs.draft_logits.shape)} dtype={outputs.draft_logits.dtype}", flush=True)
        loss = compute_dspark_loss(
            outputs=outputs,
            loss_decay_gamma=self.args.model.loss_decay_gamma,
            ce_loss_alpha=float(self.args.model.ce_loss_alpha),
            l1_loss_alpha=float(self.args.model.l1_loss_alpha),
            confidence_head_alpha=float(self.args.model.confidence_head_alpha),
        )
        if _dbg:
            print(f"[dbg]   after_loss={g():.2f}GB (loss_peak={torch.cuda.max_memory_allocated()/1e9:.2f})", flush=True)
        return loss


class Gemma4DSparkTrainer(Qwen3DSparkTrainer):
    def _build_draft_model(self, *, target_config, model_args):
        draft_config = build_gemma4_draft_config(
            target_config=target_config,
            model_args=model_args,
        )
        return Gemma4DSparkModel(draft_config)


class Qwen35DSparkTrainer(Qwen3DSparkTrainer):
    def _build_draft_model(self, *, target_config, model_args):
        draft_config = build_qwen35_draft_config(
            target_config=target_config,
            model_args=model_args,
        )
        return Qwen35DSparkModel(draft_config)
