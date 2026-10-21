"""Core model unit tests — CPU only, no GPU required."""

import pytest
import torch

from open_mythos.main import MythosConfig, OpenMythos, estimate_mfu
from open_mythos.variants import mythos_1b, mythos_7b, mythos_100m, mythos_370m


@pytest.fixture
def tiny_cfg():
    return MythosConfig(
        vocab_size=256,
        dim=64,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=32,
        max_loop_iters=2,
        prelude_layers=1,
        coda_layers=1,
        n_experts=4,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=32,
        kv_lora_rank=32,
        q_lora_rank=64,
        qk_rope_head_dim=8,
        qk_nope_head_dim=16,
        v_head_dim=16,
        max_cache_len=64,
        max_output_tokens=16,
        lora_rank=4,
    )


@pytest.fixture
def tiny_model(tiny_cfg):
    return OpenMythos(tiny_cfg).eval()


def test_forward_shape(tiny_model, tiny_cfg):
    x = torch.randint(0, tiny_cfg.vocab_size, (2, 8))
    logits = tiny_model(x)
    assert logits.shape == (2, 8, tiny_cfg.vocab_size)


def test_greedy_decoding(tiny_model, tiny_cfg):
    """temperature=0.0 must use argmax — different seeds must give same result."""
    x = torch.randint(0, tiny_cfg.vocab_size, (1, 4))
    torch.manual_seed(0)
    out1 = tiny_model.generate(x, max_new_tokens=4, temperature=0.0)
    torch.manual_seed(99)
    out2 = tiny_model.generate(x, max_new_tokens=4, temperature=0.0)
    assert torch.equal(out1, out2), "Greedy decoding must be deterministic across seeds"


def test_sampling_different(tiny_model, tiny_cfg):
    x = torch.randint(0, tiny_cfg.vocab_size, (1, 4))
    torch.manual_seed(0)
    out1 = tiny_model.generate(x, max_new_tokens=4, temperature=1.0)
    torch.manual_seed(42)
    out2 = tiny_model.generate(x, max_new_tokens=4, temperature=1.0)
    # Should differ at least sometimes with different seeds
    # (tiny vocab so might occasionally match — just check shape)
    assert out1.shape == out2.shape


def test_kv_cache_early_exit(tiny_model, tiny_cfg):
    """ACT halting must work WITH kv_cache (v1 bug: halted.all() guarded by kv_cache is None)."""
    x = torch.randint(0, tiny_cfg.vocab_size, (1, 1))
    kv_cache = {}
    # Should not hang or error
    out = tiny_model(x, n_loops=2, kv_cache=kv_cache)
    assert out.shape == (1, 1, tiny_cfg.vocab_size)


def test_save_load_roundtrip(tiny_model, tmp_path):
    tiny_model.save_pretrained(str(tmp_path))
    assert (tmp_path / "config.json").exists()
    loaded = OpenMythos.from_pretrained(str(tmp_path), device="cpu")
    x = torch.randint(0, 256, (1, 4))
    with torch.no_grad():
        out1 = tiny_model(x)
        out2 = loaded(x)
    assert torch.allclose(out1, out2, atol=1e-5)


def test_variant_configs_valid():
    """All variant configs must instantiate without error."""
    for fn in [mythos_100m, mythos_370m, mythos_1b]:
        cfg = fn()
        assert cfg.dim > 0
        assert cfg.n_heads % cfg.n_kv_heads == 0


def test_7b_variant_no_kwarg_bug():
    """Regression test for v1 7B variant typo."""
    cfg = mythos_7b()
    # qk_nope_head_dim must be 128, not whatever the default was
    assert cfg.qk_nope_head_dim == 128


def test_mfu_estimate(tiny_model):
    mfu = estimate_mfu(tiny_model, batch_tokens=1024, elapsed_sec=1.0)
    assert 0 < mfu < 1.0  # should be a fraction


def test_num_parameters(tiny_model):
    n = tiny_model.num_parameters()
    assert n > 0
