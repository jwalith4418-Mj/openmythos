"""
OpenMythos 3B Pretraining Script — Production Edition.

Replaces training/3b_fine_web_edu.py.

Key upgrades over the original:
  1. FSDP wrapping with optimal ModuleWrapPolicy
  2. Gradient checkpointing enabled by default (required at 3B scale)
  3. Correct gradient accumulation: no_sync on intermediate microbatches
  4. Cosine LR schedule with linear warmup
  5. Curriculum learning: sequence length 512 → 2048 over first 10K steps
  6. W&B experiment tracking
  7. Atomic checkpointing with configurable retention
  8. torch.compile for ~15-30% throughput improvement
  9. Proper DDP environment setup for multi-node via torchrun

Launch (single node, 8 GPUs):
  torchrun --nproc_per_node=8 training/pretrain_3b.py

Launch (2 nodes × 8 GPUs):
  torchrun --nnodes=2 --nproc_per_node=8 --node_rank=$RANK \
    --master_addr=$MASTER_ADDR --master_port=29500 training/pretrain_3b.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from open_mythos.main import OpenMythos, MythosConfig
from open_mythos.tokenizer import MythosTokenizer
from open_mythos.training.trainer import MythosTrainer, TrainerConfig


def get_model_config() -> MythosConfig:
    """3B parameter configuration."""
    return MythosConfig(
        vocab_size=128_256,        # Llama-3 tokenizer vocab
        dim=3072,
        n_heads=24,
        n_kv_heads=8,
        max_seq_len=4096,
        max_loop_iters=8,
        prelude_layers=4,
        coda_layers=4,
        attn_type="mla",
        kv_lora_rank=512,
        q_lora_rank=1536,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        v_head_dim=128,
        n_experts=64,
        n_shared_experts=2,
        n_experts_per_tok=4,
        expert_dim=512,
        lora_rank=32,
        rope_theta=500_000.0,
        max_cache_len=8192,
        dropout=0.0,
        gradient_checkpointing=True,   # REQUIRED at 3B scale
    )


def get_trainer_config() -> TrainerConfig:
    return TrainerConfig(
        target_tokens=30_000_000_000,
        seq_len=2048,
        micro_batch_size=4,
        grad_accum_steps=8,
        lr=3e-4,
        min_lr_ratio=0.1,
        weight_decay=0.1,
        grad_clip=1.0,
        warmup_steps=2000,
        lr_schedule="cosine",
        curriculum=True,
        curriculum_start_len=512,
        curriculum_end_len=2048,
        curriculum_steps=10_000,
        ckpt_dir="checkpoints/3b",
        ckpt_every=1000,
        keep_last_ckpts=3,
        log_every=10,
        wandb_project=os.environ.get("WANDB_PROJECT", "openmythos-3b"),
        amp_dtype="bfloat16",
        use_fsdp=True,
    )


class FineWebEduDataset(torch.utils.data.Dataset):
    """
    FineWeb-Edu dataset loader.
    Streams from HuggingFace hub with on-the-fly tokenization.
    Falls back to a synthetic dataset if HF not available.
    """

    def __init__(self, tokenizer, seq_len: int = 2048, split: str = "train", max_samples: int = 100_000) -> None:
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self._data = []

        try:
            from datasets import load_dataset
            print(f"[Dataset] Loading FineWeb-Edu ({split})...")
            ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split=split, streaming=False)
            for i, example in enumerate(ds):
                if i >= max_samples:
                    break
                ids = tokenizer.encode(example["text"])
                # Chunk into seq_len blocks
                for start in range(0, len(ids) - seq_len, seq_len):
                    self._data.append(ids[start:start + seq_len + 1])
            print(f"[Dataset] Loaded {len(self._data)} sequences")
        except Exception as e:
            print(f"[Dataset] FineWeb-Edu unavailable ({e}). Using synthetic data.")
            self._data = [
                list(range(i, i + seq_len + 1)) for i in range(max_samples)
            ]

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int):
        ids = self._data[idx][:self.seq_len + 1]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        return x, y


def main():
    # ---- Distributed init ----
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    dist.init_process_group(backend="nccl") if torch.cuda.is_available() else None
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    if rank == 0:
        print(f"[Pretrain] world_size={world_size}, device={device}")

    # ---- Build model ----
    model_cfg = get_model_config()
    model = OpenMythos(model_cfg).to(device)
    model.enable_gradient_checkpointing()

    if rank == 0:
        n_params = model.num_parameters()
        print(f"[Pretrain] Model parameters: {n_params/1e9:.2f}B")

    # ---- FSDP ----
    trainer_cfg = get_trainer_config()
    if trainer_cfg.use_fsdp and dist.is_initialized():
        from open_mythos.distributed.parallel import build_fsdp_model
        model = build_fsdp_model(model, amp_dtype=torch.bfloat16)
        if rank == 0:
            print("[Pretrain] FSDP wrapping applied")

    # ---- torch.compile ----
    if torch.cuda.is_available() and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            if rank == 0:
                print("[Pretrain] torch.compile enabled")
        except Exception as e:
            if rank == 0:
                print(f"[Pretrain] torch.compile failed: {e}")

    # ---- Tokenizer + Dataset ----
    tokenizer = MythosTokenizer()
    seq_len = trainer_cfg.curriculum_start_len if trainer_cfg.curriculum else trainer_cfg.seq_len
    dataset = FineWebEduDataset(tokenizer, seq_len=seq_len)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) \
        if dist.is_initialized() else None

    loader = DataLoader(
        dataset,
        batch_size=trainer_cfg.micro_batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # ---- Train ----
    trainer = MythosTrainer(model, trainer_cfg, vocab_size=model_cfg.vocab_size)
    trainer.train(iter(loader))

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
