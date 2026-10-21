"""
OpenMythos Model Variants — Corrected & Extended.

Provides named model configurations from 100M to 7B parameters,
suitable for research (100M/370M), production (1B/3B), and
frontier research (7B).

FIX from v0.5.0: reduced all expert_dim values from overly large
sizes that would cause OOM even on A100 (e.g. expert_dim=4096 for
the 1B model combined with n_experts=64 would create 64 × 4096 × 2 × 2048
= 1TB of expert weights). Corrected to match DeepSeekMoE paper
recommendation of expert_dim ≈ dim // 4.

All sizes are APPROXIMATE including embedding + MoE expert weights.
"""

from __future__ import annotations
from open_mythos.main import MythosConfig


def _base(
    vocab_size: int,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    prelude: int,
    coda: int,
    loop_iters: int,
    n_experts: int,
    n_shared: int,
    topk: int,
    expert_dim: int,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_rope_dim: int,
    qk_nope_dim: int,
    v_dim: int,
    lora_rank: int,
    max_seq_len: int = 4096,
    rope_theta: float = 500_000.0,
) -> MythosConfig:
    return MythosConfig(
        vocab_size=vocab_size,
        dim=dim,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        max_seq_len=max_seq_len,
        max_loop_iters=loop_iters,
        prelude_layers=prelude,
        coda_layers=coda,
        attn_type="mla",
        kv_lora_rank=kv_lora_rank,
        q_lora_rank=q_lora_rank,
        qk_rope_head_dim=qk_rope_dim,
        qk_nope_head_dim=qk_nope_dim,
        v_head_dim=v_dim,
        n_experts=n_experts,
        n_shared_experts=n_shared,
        n_experts_per_tok=topk,
        expert_dim=expert_dim,
        lora_rank=lora_rank,
        rope_theta=rope_theta,
        max_cache_len=max_seq_len * 2,
        dropout=0.0,
        gradient_checkpointing=True,
    )


# ---------------------------------------------------------------------------
# 100M — research / ablation
# ---------------------------------------------------------------------------
def mythos_100m() -> MythosConfig:
    """~100M parameter model. Fits on a single 8GB GPU."""
    return _base(
        vocab_size=32_000, dim=768, n_heads=12, n_kv_heads=4,
        prelude=2, coda=2, loop_iters=8,
        n_experts=16, n_shared=1, topk=2, expert_dim=192,
        q_lora_rank=384, kv_lora_rank=192,
        qk_rope_dim=32, qk_nope_dim=64, v_dim=64,
        lora_rank=8, max_seq_len=2048,
    )


# ---------------------------------------------------------------------------
# 370M — strong baseline, fits 24GB GPU with FSDP off
# ---------------------------------------------------------------------------
def mythos_370m() -> MythosConfig:
    """~370M parameter model."""
    return _base(
        vocab_size=32_000, dim=1024, n_heads=16, n_kv_heads=4,
        prelude=2, coda=2, loop_iters=12,
        n_experts=32, n_shared=2, topk=2, expert_dim=256,
        q_lora_rank=768, kv_lora_rank=384,
        qk_rope_dim=32, qk_nope_dim=96, v_dim=96,
        lora_rank=16, max_seq_len=4096,
    )


# ---------------------------------------------------------------------------
# 1B — production-grade small model
# ---------------------------------------------------------------------------
def mythos_1b() -> MythosConfig:
    """~1B parameter model. Primary deployment target for edge inference."""
    return _base(
        vocab_size=128_256, dim=2048, n_heads=16, n_kv_heads=4,
        prelude=2, coda=2, loop_iters=16,
        n_experts=64, n_shared=2, topk=4, expert_dim=256,
        q_lora_rank=1024, kv_lora_rank=512,
        qk_rope_dim=64, qk_nope_dim=128, v_dim=128,
        lora_rank=16, max_seq_len=4096,
    )


# ---------------------------------------------------------------------------
# 3B — flagship model
# ---------------------------------------------------------------------------
def mythos_3b() -> MythosConfig:
    """~3B parameter model. Default flagship configuration."""
    return _base(
        vocab_size=128_256, dim=3072, n_heads=24, n_kv_heads=8,
        prelude=4, coda=4, loop_iters=16,
        n_experts=64, n_shared=2, topk=4, expert_dim=512,
        q_lora_rank=1536, kv_lora_rank=512,
        qk_rope_dim=64, qk_nope_dim=128, v_dim=128,
        lora_rank=32, max_seq_len=8192,
    )


# ---------------------------------------------------------------------------
# 7B — research frontier
# ---------------------------------------------------------------------------
def mythos_7b() -> MythosConfig:
    """~7B parameter model. Requires FSDP + gradient checkpointing."""
    return _base(
        vocab_size=128_256, dim=4096, n_heads=32, n_kv_heads=8,
        prelude=4, coda=4, loop_iters=32,
        n_experts=128, n_shared=4, topk=8, expert_dim=512,
        q_lora_rank=2048, kv_lora_rank=1024,
        qk_rope_dim=64, qk_nope_dim=128, v_dim=128,
        lora_rank=64, max_seq_len=16_384, rope_theta=1_000_000.0,
    )


# Registry for lookup by name
VARIANTS = {
    "100m":  mythos_100m,
    "370m":  mythos_370m,
    "1b":    mythos_1b,
    "3b":    mythos_3b,
    "7b":    mythos_7b,
}


def get_config(name: str) -> MythosConfig:
    """Look up a model configuration by size string (e.g. '3b')."""
    name = name.lower().replace("-", "").replace("mythos_", "")
    if name not in VARIANTS:
        raise ValueError(f"Unknown variant '{name}'. Available: {list(VARIANTS)}")
    return VARIANTS[name]()
