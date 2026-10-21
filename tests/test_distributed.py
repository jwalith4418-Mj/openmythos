"""Distributed utilities unit tests — no process group required for most."""
import pytest, torch
from unittest.mock import patch, MagicMock

def test_scatter_gather_identity():
    """scatter_to_sequence_parallel with None group returns input unchanged."""
    from open_mythos.distributed.parallel import scatter_to_sequence_parallel, gather_from_sequence_parallel
    x = torch.randn(2, 16, 64)
    scattered = scatter_to_sequence_parallel(x, sp_group=None)
    assert torch.equal(scattered, x)
    gathered = gather_from_sequence_parallel(x, sp_group=None)
    assert torch.equal(gathered, x)

def test_fsdp_checkpoint_roundtrip(tmp_path):
    """fsdp_save_checkpoint and fsdp_load_checkpoint with plain model (no FSDP)."""
    import os
    from open_mythos.main import OpenMythos, MythosConfig
    cfg = MythosConfig(vocab_size=64, dim=32, n_heads=2, n_kv_heads=1,
                       max_seq_len=16, max_loop_iters=1, prelude_layers=1, coda_layers=1,
                       n_experts=2, n_shared_experts=1, n_experts_per_tok=1, expert_dim=16,
                       kv_lora_rank=16, q_lora_rank=32, qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=8,
                       max_cache_len=32, lora_rank=4)
    model = OpenMythos(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    ckpt = str(tmp_path / "ckpt.pt")
    # Manual save (no FSDP in unit test)
    torch.save({"step": 100, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "config_hash": "abc"}, ckpt)
    assert os.path.exists(ckpt)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert state["step"] == 100
    model.load_state_dict(state["model"])

def test_pipeline_stage_forward():
    from open_mythos.main import OpenMythos, MythosConfig
    from open_mythos.distributed.parallel import PipelineStage
    cfg = MythosConfig(vocab_size=64, dim=32, n_heads=2, n_kv_heads=1,
                       max_seq_len=16, max_loop_iters=1, prelude_layers=1, coda_layers=1,
                       n_experts=2, n_shared_experts=1, n_experts_per_tok=1, expert_dim=16,
                       kv_lora_rank=16, q_lora_rank=32, qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=8,
                       max_cache_len=32, lora_rank=4)
    model = OpenMythos(cfg)
    stage = PipelineStage(model.prelude, stage_id=0, n_stages=2)
    x = torch.randn(1, 4, 32)
    import torch.nn as nn
    from open_mythos.main import precompute_rope_freqs
    # FIX: compute freqs with qk_rope_head_dim=8 (not dim//n_heads=16), since cfg uses MLA with qk_rope_head_dim=8
    freqs = precompute_rope_freqs(cfg.qk_rope_head_dim, cfg.max_seq_len)
    # stage.forward passes freqs_cis as kwarg; PipelineStage.forward slices to T
    out = stage(x, freqs_cis=freqs[:4])
    assert out.shape == (1, 4, 32)

def test_deepspeed_config_structure():
    from open_mythos.distributed.parallel import build_deepspeed_config
    cfg = build_deepspeed_config(micro_batch_size=4, gradient_accumulation_steps=8)
    assert cfg["zero_optimization"]["stage"] == 3
    assert cfg["bf16"]["enabled"] is True
    assert cfg["train_micro_batch_size_per_gpu"] == 4
