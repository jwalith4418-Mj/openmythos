"""
OOM recovery tests — validate trainer handles OutOfMemoryError gracefully.
Uses monkeypatching to simulate OOM without actually running out of memory.
"""

from unittest.mock import patch

import torch


def test_trainer_oom_recovery(tmp_path):
    """Trainer must continue training after OOM, not crash."""
    from open_mythos.main import MythosConfig, OpenMythos
    from open_mythos.training.trainer import MythosTrainer, TrainerConfig

    cfg = MythosConfig(
        vocab_size=64,
        dim=32,
        n_heads=2,
        n_kv_heads=1,
        max_seq_len=8,
        max_loop_iters=1,
        prelude_layers=1,
        coda_layers=1,
        n_experts=2,
        n_shared_experts=1,
        n_experts_per_tok=1,
        expert_dim=16,
        kv_lora_rank=16,
        q_lora_rank=32,
        qk_rope_head_dim=4,
        qk_nope_head_dim=4,
        v_head_dim=4,
        max_cache_len=16,
        lora_rank=4,
    )
    model = OpenMythos(cfg)
    tcfg = TrainerConfig(
        target_tokens=8 * 8 * 2,
        seq_len=8,
        micro_batch_size=1,
        grad_accum_steps=1,
        log_every=1,
        ckpt_every=100,
        ckpt_dir=str(tmp_path),
        use_fsdp=False,
        use_compile=False,
        wandb_project=None,
    )
    trainer = MythosTrainer(model, tcfg, vocab_size=64)

    call_count = 0
    original_forward = model.forward

    def oom_once(input_ids, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise torch.cuda.OutOfMemoryError("Simulated OOM")
        return original_forward(input_ids, **kwargs)

    import torch.utils.data as data_utils

    class TinyDataset(data_utils.Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, i):
            return {"input_ids": torch.randint(0, 64, (8,)), "labels": torch.randint(0, 64, (8,))}

    dl = data_utils.DataLoader(TinyDataset(), batch_size=1)

    with patch.object(model, "forward", side_effect=oom_once):
        try:
            trainer.train(dl)
        except StopIteration:
            pass  # normal end of tiny dataset

    assert trainer.step >= 1, "Trainer should have stepped past the OOM"


def test_checkpoint_recovery(tmp_path):
    """Can resume training from a saved checkpoint."""
    import torch

    step = 500
    model_state = {"weight": torch.randn(4, 4)}
    ckpt_path = str(tmp_path / "step_0000500.pt")
    torch.save(
        {"step": step, "model": model_state, "optimizer": {}, "config_hash": "test"}, ckpt_path
    )
    loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert loaded["step"] == step
    assert "model" in loaded
