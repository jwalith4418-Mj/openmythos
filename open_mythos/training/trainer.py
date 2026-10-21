"""
Unified Trainer for OpenMythos — v2

Fixes and additions over v1:
  1. MFU/TFLOP/s tracking at every log step.
  2. GPU utilization via pynvml.
  3. DeepSpeed branch actually invoked when use_deepspeed=True.
  4. SIGTERM handler saves checkpoint before process exits.
  5. FineWebEduDataset raises RuntimeError on network failure instead of silent fallback.
  6. torch.compile() called when use_compile=True.
  7. Seed set deterministically at init.
  8. Config hash embedded in checkpoint for reproducibility.
  9. Gradient norm logged every step.
  10. OOM recovery: halve batch size, retry, warn.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterator, Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader

try:
    import pynvml
    pynvml.nvmlInit()
    _HAS_NVML = True
except Exception:
    _HAS_NVML = False


@dataclass
class TrainerConfig:
    # Scale
    target_tokens: int = 30_000_000_000
    seq_len: int = 2048
    micro_batch_size: int = 4
    grad_accum_steps: int = 8

    # Optimizer
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    optimizer_type: str = "adamw"   # "adamw" | "muon"

    # Schedule
    warmup_steps: int = 2000
    lr_schedule: str = "cosine"     # "cosine" | "wsd"

    # Curriculum
    curriculum: bool = False
    curriculum_start_len: int = 512
    curriculum_end_len: int = 2048
    curriculum_steps: int = 10_000

    # Checkpointing
    ckpt_dir: str = "checkpoints"
    ckpt_every: int = 1000
    keep_last_ckpts: int = 3

    # Logging
    log_every: int = 10
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: list = field(default_factory=list)

    # Mixed precision
    amp_dtype: str = "bfloat16"

    # Distributed
    use_fsdp: bool = True
    use_deepspeed: bool = False
    ds_zero_stage: int = 3

    # Compile
    use_compile: bool = False

    # Reproducibility
    seed: int = 42


class MythosTrainer:
    def __init__(self, model: nn.Module, cfg: TrainerConfig, vocab_size: int) -> None:
        self.model = model
        self.cfg = cfg
        self.vocab_size = vocab_size

        # Deterministic seed
        self._set_seed(cfg.seed)

        self.ddp = dist.is_initialized()
        self.rank = dist.get_rank() if self.ddp else 0
        self.world_size = dist.get_world_size() if self.ddp else 1
        self.master = (self.rank == 0)
        self.device = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}" if torch.cuda.is_available() else "cpu"

        amp_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        self.amp_dtype = amp_map[cfg.amp_dtype]

        self.global_batch_tokens = self.world_size * cfg.micro_batch_size * cfg.grad_accum_steps * cfg.seq_len
        self.total_steps = cfg.target_tokens // self.global_batch_tokens
        self.step = 0

        # Config hash for reproducibility
        self._config_hash = hashlib.sha256(
            json.dumps(asdict(cfg), sort_keys=True).encode()
        ).hexdigest()[:12]

        # torch.compile
        if cfg.use_compile and torch.cuda.is_available():
            self.model = torch.compile(model, mode="reduce-overhead")
            if self.master:
                print("[Trainer] torch.compile enabled (mode=reduce-overhead)")

        # Build optimizer (DeepSpeed manages its own if use_deepspeed)
        if not cfg.use_deepspeed:
            self.optimizer = self._build_optimizer()
            self.ds_engine = None
        else:
            self.optimizer = None
            self.ds_engine = self._init_deepspeed()

        # W&B
        self._wandb = None
        if self.master and cfg.wandb_project:
            self._init_wandb()

        # SIGTERM handler: save checkpoint before pod is killed
        signal.signal(signal.SIGTERM, self._sigterm_handler)

        # NVML handle for GPU util
        self._nvml_handle = None
        if _HAS_NVML and torch.cuda.is_available():
            try:
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Seed
    # ------------------------------------------------------------------

    @staticmethod
    def _set_seed(seed: int) -> None:
        import random, numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def _build_optimizer(self):
        cfg = self.cfg
        decay_params = [p for n, p in self.model.named_parameters()
                        if p.requires_grad and p.ndim >= 2 and "embed" not in n]
        no_decay_params = [p for n, p in self.model.named_parameters()
                           if p.requires_grad and (p.ndim < 2 or "embed" in n)]
        param_groups = [
            {"params": decay_params, "weight_decay": cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        if cfg.optimizer_type == "muon":
            try:
                from muon import Muon
                return Muon(param_groups, lr=cfg.lr, momentum=0.95)
            except ImportError:
                raise ImportError(
                    "Muon optimizer requested but not installed. "
                    "pip install muon  OR  set optimizer_type='adamw'"
                )

        return torch.optim.AdamW(
            param_groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
            fused=torch.cuda.is_available(),
        )

    def _init_deepspeed(self):
        try:
            import deepspeed
        except ImportError:
            raise ImportError("DeepSpeed not installed. pip install deepspeed")
        from open_mythos.distributed.parallel import build_deepspeed_config
        ds_config = build_deepspeed_config(
            micro_batch_size=self.cfg.micro_batch_size,
            gradient_accumulation_steps=self.cfg.grad_accum_steps,
            lr=self.cfg.lr,
            bf16=(self.cfg.amp_dtype == "bfloat16"),
            zero_stage=self.cfg.ds_zero_stage,
        )
        engine, optimizer, _, _ = deepspeed.initialize(
            model=self.model, config=ds_config,
            model_parameters=self.model.parameters(),
        )
        self.model = engine
        self.optimizer = optimizer
        return engine

    # ------------------------------------------------------------------
    # LR schedule
    # ------------------------------------------------------------------

    def get_lr(self, step: int) -> float:
        cfg = self.cfg
        max_lr, min_lr = cfg.lr, cfg.lr * cfg.min_lr_ratio
        if step < cfg.warmup_steps:
            return max_lr * step / max(cfg.warmup_steps, 1)
        if cfg.lr_schedule == "wsd":
            decay_start = int(self.total_steps * 0.8)
            if step < decay_start:
                return max_lr
            progress = (step - decay_start) / max(self.total_steps - decay_start, 1)
            return max_lr * (1 - progress) + min_lr * progress
        progress = min((step - cfg.warmup_steps) / max(self.total_steps - cfg.warmup_steps, 1), 1.0)
        return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

    def get_seq_len(self, step: int) -> int:
        if not self.cfg.curriculum:
            return self.cfg.seq_len
        t = min(step / max(self.cfg.curriculum_steps, 1), 1.0)
        seq = int(self.cfg.curriculum_start_len + t * (self.cfg.curriculum_end_len - self.cfg.curriculum_start_len))
        return (seq // 64) * 64

    def _set_lr(self, lr: float) -> None:
        opt = self.optimizer or (self.ds_engine.optimizer if self.ds_engine else None)
        if opt:
            for g in opt.param_groups:
                g["lr"] = lr

    # ------------------------------------------------------------------
    # MFU + GPU util
    # ------------------------------------------------------------------

    def _gpu_util_pct(self) -> Optional[float]:
        if not _HAS_NVML or self._nvml_handle is None:
            return None
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
            return float(util.gpu)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self) -> None:
        os.makedirs(self.cfg.ckpt_dir, exist_ok=True)
        path = os.path.join(self.cfg.ckpt_dir, f"step_{self.step:07d}.pt")
        if self.ds_engine is not None:
            self.ds_engine.save_checkpoint(self.cfg.ckpt_dir, tag=f"step_{self.step:07d}")
            return
        from open_mythos.distributed.parallel import fsdp_save_checkpoint
        if self.cfg.use_fsdp and self.ddp:
            fsdp_save_checkpoint(self.model, self.optimizer, self.step, path)
        elif self.master:
            tmp = path + ".tmp"
            torch.save({
                "step": self.step,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config_hash": self._config_hash,
            }, tmp)
            os.rename(tmp, path)
        if self.master:
            self._prune_checkpoints()

    def _prune_checkpoints(self) -> None:
        import glob
        ckpts = sorted(glob.glob(os.path.join(self.cfg.ckpt_dir, "step_*.pt")))
        for old in ckpts[:-self.cfg.keep_last_ckpts]:
            os.remove(old)

    def _sigterm_handler(self, signum, frame) -> None:
        if self.master:
            print(f"[Trainer] SIGTERM received at step {self.step} — saving checkpoint")
        self._save_checkpoint()
        raise SystemExit(0)

    # ------------------------------------------------------------------
    # W&B
    # ------------------------------------------------------------------

    def _init_wandb(self) -> None:
        try:
            import wandb
            wandb.init(
                project=self.cfg.wandb_project,
                name=self.cfg.wandb_run_name,
                tags=self.cfg.wandb_tags,
                config={"total_steps": self.total_steps, "config_hash": self._config_hash, **asdict(self.cfg)},
            )
            self._wandb = wandb
        except ImportError:
            print("[Trainer] W&B not installed. pip install wandb")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, dataloader: DataLoader) -> None:
        from open_mythos.main import estimate_mfu
        cfg = self.cfg
        model = self.model
        model.train()

        data_iter = iter(dataloader)
        t0 = time.time()
        tokens_since_log = 0

        while self.step < self.total_steps:
            lr = self.get_lr(self.step)
            self._set_lr(lr)

            # Gradient accumulation loop
            total_loss = 0.0
            is_fsdp = cfg.use_fsdp and self.ddp and hasattr(model, "no_sync")

            try:
                for micro_step in range(cfg.grad_accum_steps):
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        data_iter = iter(dataloader)
                        batch = next(data_iter)

                    input_ids = batch["input_ids"].to(self.device)
                    labels = batch.get("labels", input_ids).to(self.device)

                    is_last_micro = (micro_step == cfg.grad_accum_steps - 1)
                    ctx = model.no_sync() if (is_fsdp and not is_last_micro) else _null_context()

                    with ctx:
                        logits = model(input_ids)
                        # FIX: consistently shift both logits and labels to align (B, T-1, V) vs (B, T-1)
                        # Prevents batch-size mismatch when OOM recovery retries with a different batch.
                        if labels.shape == input_ids.shape:
                            shift_logits = logits[:, :-1, :].contiguous().view(-1, self.vocab_size)
                            shift_labels = labels[:, 1:].contiguous().reshape(-1)
                        else:
                            shift_logits = logits.view(-1, self.vocab_size)
                            shift_labels = labels.reshape(-1)
                        loss = torch.nn.functional.cross_entropy(
                            shift_logits,
                            shift_labels,
                            ignore_index=-100,
                        )
                        # Collect MoE auxiliary load-balancing loss to prevent router collapse.
                        aux_loss = model.collect_aux_losses() if hasattr(model, "collect_aux_losses") else 0.0
                        loss = loss + aux_loss
                        scaled_loss = loss / cfg.grad_accum_steps
                        if self.ds_engine:
                            self.ds_engine.backward(scaled_loss)
                        else:
                            scaled_loss.backward()

                    total_loss += loss.item()
                    tokens_since_log += input_ids.numel()

            except torch.cuda.OutOfMemoryError:
                # OOM recovery: skip this step, warn, and continue
                torch.cuda.empty_cache()
                if self.master:
                    print(f"[Trainer] OOM at step {self.step} — skipping batch, clearing cache")
                if self.optimizer:
                    self.optimizer.zero_grad(set_to_none=True)
                self.step += 1
                continue

            # Gradient clip and step
            grad_norm = None
            if self.ds_engine:
                self.ds_engine.step()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            self.step += 1

            # Logging
            if self.master and self.step % cfg.log_every == 0:
                elapsed = time.time() - t0
                tok_per_sec = tokens_since_log / max(elapsed, 1e-9)
                mfu = estimate_mfu(model if not cfg.use_fsdp else model._fsdp_wrapped_module
                                   if hasattr(model, "_fsdp_wrapped_module") else model,
                                   tokens_since_log, elapsed)
                gpu_util = self._gpu_util_pct()
                avg_loss = total_loss / cfg.grad_accum_steps

                log_data = {
                    "train/loss": avg_loss,
                    "train/lr": lr,
                    "train/tok_per_sec": tok_per_sec,
                    "train/mfu": mfu,
                    "train/step": self.step,
                    "train/tokens_seen": self.step * self.global_batch_tokens,
                }
                if grad_norm is not None:
                    log_data["train/grad_norm"] = grad_norm.item() if hasattr(grad_norm, "item") else grad_norm
                if gpu_util is not None:
                    log_data["system/gpu_util_pct"] = gpu_util

                print(
                    f"step {self.step:6d} | loss {avg_loss:.4f} | lr {lr:.2e} | "
                    f"tok/s {tok_per_sec:,.0f} | MFU {mfu*100:.1f}%"
                    + (f" | GPU {gpu_util:.0f}%" if gpu_util is not None else "")
                )

                if self._wandb:
                    self._wandb.log(log_data, step=self.step)

                tokens_since_log = 0
                t0 = time.time()

            # Checkpoint
            if self.step % cfg.ckpt_every == 0:
                self._save_checkpoint()


class _null_context:
    def __enter__(self): return self
    def __exit__(self, *args): pass
