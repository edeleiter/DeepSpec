import os

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR as _CosineAnnealingLR
from torch.optim.lr_scheduler import LRScheduler as _LRScheduler


class TwoStageScheduler(_LRScheduler):
    def __init__(self, optimizer, after_scheduler: _LRScheduler, last_epoch=-1):
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer, last_epoch)

    def state_dict(self):
        state_dict = {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }
        if isinstance(state_dict["after_scheduler"], _LRScheduler):
            state_dict["after_scheduler_type"] = type(
                state_dict["after_scheduler"]
            ).__name__
            state_dict["after_scheduler_dict"] = state_dict[
                "after_scheduler"
            ].state_dict()
            del state_dict["after_scheduler"]
        else:
            raise NotImplementedError()
        return state_dict

    def load_state_dict(self, state_dict):
        self.after_scheduler.load_state_dict(state_dict["after_scheduler_dict"])
        state_dict = {
            key: value
            for key, value in state_dict.items()
            if key not in ("after_scheduler_type", "after_scheduler_dict")
        }
        super().load_state_dict(state_dict)


class WarmupScheduler(TwoStageScheduler):
    def __init__(self, optimizer, warmup_epochs, after_scheduler, last_epoch=-1):
        self.warmup_epochs = int(warmup_epochs)
        super().__init__(optimizer, after_scheduler, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.warmup_epochs:
            if not self.finished:
                self.after_scheduler.base_lrs = self.base_lrs
                self.finished = True
            return self.after_scheduler.get_lr()

        return [(self.last_epoch + 1) / self.warmup_epochs * lr for lr in self.base_lrs]

    def step(self, epoch=None):
        if self.finished:
            if epoch is None:
                self.after_scheduler.step(None)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                self.after_scheduler.step(epoch - self.warmup_epochs)
                self._last_lr = self.after_scheduler.get_last_lr()
        else:
            return super().step(epoch)


class CosineAnnealingWarmupLR(WarmupScheduler):
    def __init__(
        self,
        optimizer,
        total_steps: int,
        warmup_steps: int = 0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ):
        base_scheduler = _CosineAnnealingLR(
            optimizer,
            total_steps - warmup_steps,
            eta_min=eta_min,
            last_epoch=last_epoch,
        )
        super().__init__(optimizer, warmup_steps, base_scheduler, last_epoch=last_epoch)


class BF16Optimizer:
    # Adapted from SpecForge/specforge/optimizer.py:BF16Optimizer.
    def __init__(
        self,
        model,
        lr,
        total_steps,
        warmup_ratio,
        weight_decay=0.0,
    ):
        self.model = model
        # Iterate named_parameters() (identical order to parameters(), names
        # attached) so the MuonAdam split can select by name. With Muon off the
        # ordering is unchanged, keeping the OFF path byte-identical to before.
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self.model_params = [p for _, p in named]
        self.fp32_params = [
            p.detach().clone().to(torch.float32) for p in self.model_params
        ]
        for param in self.fp32_params:
            param.requires_grad = True

        # Muon handles the draft's 2D hidden weight matrices (attention/MLP
        # projections + the fc fusion, ~87% of trainable params); everything else
        # -- 1D norm gains, the markov head, the vocab-sized markov_w2 head, and
        # the confidence head -- stays on AdamW. Default OFF (DSPARK_MUON unset)
        # puts ALL params on Adam, byte-identical to the pre-Muon optimizer.
        use_muon = bool(os.environ.get("DSPARK_MUON"))
        if use_muon:
            def _is_muon(name, param):
                return param.ndim == 2 and not any(
                    key in name
                    for key in ("markov", "lm_head", "embed_tokens", "confidence_head")
                )
            muon_idx = [i for i, (n, p) in enumerate(named) if _is_muon(n, p)]
        else:
            muon_idx = []
        muon_set = set(muon_idx)
        adam_fp32 = [self.fp32_params[i] for i in range(len(named)) if i not in muon_set]
        muon_fp32 = [self.fp32_params[i] for i in muon_idx]

        # --- Adam group (all params when Muon off). 8-bit Adam quantizes the
        # optimizer moments (~8 -> ~2 bytes/param) and now governs only the ~13%
        # non-matrix params when Muon is on. Falls back to fp32 AdamW. ---
        if os.environ.get("DSPARK_ADAM8BIT"):
            import bitsandbytes as bnb

            self.adam = bnb.optim.Adam8bit(
                adam_fp32, lr=lr, weight_decay=weight_decay
            )
        else:
            self.adam = torch.optim.AdamW(
                adam_fp32, lr=lr, weight_decay=weight_decay
            )
        self.adam_scheduler = CosineAnnealingWarmupLR(
            self.adam,
            total_steps=total_steps,
            warmup_steps=int(warmup_ratio * total_steps),
        )

        # --- Muon group (optional). Shares the Adam LR via the RMS-matched update
        # scale; DSPARK_MUON_LR_SCALE is the one tuning lever, DSPARK_MUON_WD adds
        # decoupled weight decay for long runs. Its own scheduler steps in lockstep
        # with Adam's (identical cosine-warmup shape). ---
        self.muon = None
        self.muon_scheduler = None
        if use_muon and muon_fp32:
            from deepspec.utils.muon import Muon

            scale = float(os.environ.get("DSPARK_MUON_LR_SCALE", "1.0"))
            self.muon = Muon(
                muon_fp32,
                lr=lr * scale,
                weight_decay=float(os.environ.get("DSPARK_MUON_WD", "0.0")),
            )
            self.muon_scheduler = CosineAnnealingWarmupLR(
                self.muon,
                total_steps=total_steps,
                warmup_steps=int(warmup_ratio * total_steps),
            )

    def step(self):
        # Fill fp32 master grads for the FULL param list; each sub-optimizer
        # reads only its own subset, and zero_grad on each clears its subset.
        with torch.no_grad():
            for model_param, master_param in zip(self.model_params, self.fp32_params):
                master_param.grad = (
                    model_param.grad.detach().to(torch.float32)
                    if model_param.grad is not None
                    else None
                )
        self.adam.step()
        if self.muon is not None:
            self.muon.step()
        self.adam.zero_grad(set_to_none=True)
        if self.muon is not None:
            self.muon.zero_grad(set_to_none=True)
        self.adam_scheduler.step()
        if self.muon_scheduler is not None:
            self.muon_scheduler.step()
        with torch.no_grad():
            for model_param, master_param in zip(self.model_params, self.fp32_params):
                model_param.data.copy_(master_param.data.to(model_param.dtype))
                model_param.grad = None

    def state_dict(self):
        # Legacy keys (optimizer_/scheduler_state_dict) preserved so pre-Muon
        # AdamW-only checkpoints still load. fp32_params is the full, order-stable
        # master list and round-trips regardless of the muon/adam split.
        return {
            "optimizer_state_dict": self.adam.state_dict(),
            "scheduler_state_dict": self.adam_scheduler.state_dict(),
            "muon_optimizer_state_dict": (
                self.muon.state_dict() if self.muon is not None else None
            ),
            "muon_scheduler_state_dict": (
                self.muon_scheduler.state_dict() if self.muon_scheduler is not None else None
            ),
            "muon_enabled": self.muon is not None,
            "fp32_params": [param.detach().cpu() for param in self.fp32_params],
        }

    def load_state_dict(self, state_dict):
        # Resuming across an optimizer switch would misalign per-optimizer state
        # indices; fail loudly rather than silently corrupt. Muon is a fresh-run
        # experiment, not a mid-run resume.
        ckpt_muon = state_dict.get("muon_enabled", False)
        if ckpt_muon != (self.muon is not None):
            raise RuntimeError(
                f"Optimizer mismatch: checkpoint muon_enabled={ckpt_muon} but current "
                f"run muon_enabled={self.muon is not None}. Resuming across an optimizer "
                f"switch is unsupported; start a fresh run."
            )
        self.adam.load_state_dict(state_dict["optimizer_state_dict"])
        self.adam_scheduler.load_state_dict(state_dict["scheduler_state_dict"])
        if self.muon is not None:
            self.muon.load_state_dict(state_dict["muon_optimizer_state_dict"])
            self.muon_scheduler.load_state_dict(state_dict["muon_scheduler_state_dict"])
        fp32_params = state_dict["fp32_params"]
        for dst, src in zip(self.fp32_params, fp32_params):
            dst.data.copy_(src.to(dst.device))
        with torch.no_grad():
            for model_param, master_param in zip(self.model_params, self.fp32_params):
                model_param.data.copy_(master_param.data.to(model_param.dtype))

    def get_learning_rate(self):
        return self.adam.param_groups[0]["lr"]
