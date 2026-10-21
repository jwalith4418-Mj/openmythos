"""
Tests for OpenMythos v4 inference additions:
  - InferenceProfiler / Reasoning-FLOP Efficiency
  - SpeculativeDepthDecoder
"""

import pytest
import torch

from open_mythos.main import MythosConfig, OpenMythos


@pytest.fixture
def tiny_cfg():
    return MythosConfig(
        vocab_size=256,
        dim=64,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=32,
        max_loop_iters=4,
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
        act_threshold=0.99,
    )


@pytest.fixture
def tiny_model(tiny_cfg):
    return OpenMythos(tiny_cfg).eval()


class FakeTokenizer:
    eos_token_id = 2

    def encode(self, text):
        return [1] + [ord(c) % 200 for c in text[:10]] + [2]

    def decode(self, ids):
        return "".join(chr(min(i + 32, 126)) for i in ids if i > 2)


# ─── InferenceProfiler ────────────────────────────────────────────────────────


class TestInferenceProfiler:
    def test_profile_query_returns_profile(self, tiny_model):
        from open_mythos.inference.profiler import InferenceProfiler

        tok = FakeTokenizer()
        profiler = InferenceProfiler(tiny_model, tok, device="cpu", max_loops=2)
        profile = profiler.profile_query("hello", max_new_tokens=4, n_loops=2)
        assert profile.n_generated_tokens >= 1
        assert profile.latency_ms > 0
        assert profile.total_flops > 0

    def test_profile_dataset_returns_report(self, tiny_model):
        from open_mythos.inference.profiler import InferenceProfiler

        tok = FakeTokenizer()
        profiler = InferenceProfiler(tiny_model, tok, device="cpu", max_loops=2)
        prompts = ["hello", "world"]
        report = profiler.profile_dataset(prompts, max_new_tokens=4, n_loops=2)
        assert report.n_queries == 2
        assert report.mean_loop_count > 0
        assert report.mean_flops_per_token > 0
        assert report.reasoning_flop_efficiency is None  # no ground truths

    def test_rfe_computed_with_ground_truths(self, tiny_model):
        from open_mythos.inference.profiler import InferenceProfiler

        tok = FakeTokenizer()
        profiler = InferenceProfiler(tiny_model, tok, device="cpu", max_loops=2)
        prompts = ["hello"]
        truths = ["anything"]  # will likely not match → accuracy ~0
        report = profiler.profile_dataset(
            prompts,
            ground_truths=truths,
            eval_fn=lambda g, t: t in g,
            max_new_tokens=4,
            n_loops=2,
        )
        assert report.reasoning_flop_efficiency is not None
        assert report.accuracy is not None
        assert 0.0 <= report.accuracy <= 1.0

    def test_loop_count_tracked_on_recurrent_block(self, tiny_model):
        """RecurrentBlock must store _last_loop_count after forward."""
        x = torch.randint(0, 256, (1, 4))
        _ = tiny_model(x, n_loops=3)
        # Find the recurrent block and check
        lc = getattr(tiny_model.recurrent, "_last_loop_count", None)
        assert lc is not None, "_last_loop_count must be set after forward"
        assert 1 <= lc <= 3

    def test_report_save_json(self, tiny_model, tmp_path):
        from open_mythos.inference.profiler import InferenceProfiler

        tok = FakeTokenizer()
        profiler = InferenceProfiler(tiny_model, tok, device="cpu", max_loops=2)
        report = profiler.profile_dataset(["test"], max_new_tokens=2, n_loops=2)
        out = str(tmp_path / "rfe.json")
        report.save_json(out)
        import json

        d = json.load(open(out))
        assert "mean_flops_per_token" in d
        assert "mean_loop_count" in d


# ─── SpeculativeDepthDecoder ──────────────────────────────────────────────────


class TestSpeculativeDepthDecoder:
    def test_generates_correct_length(self, tiny_model):
        from open_mythos.inference.speculative import SpeculativeDepthDecoder

        tok = FakeTokenizer()
        decoder = SpeculativeDepthDecoder(
            tiny_model,
            tok,
            n_draft_loops=1,
            n_verify_loops=2,
            n_draft_tokens=2,
            device="cpu",
        )
        input_ids = torch.randint(0, 256, (1, 4))
        output = decoder.generate(input_ids, max_new_tokens=8, temperature=1.0)
        assert output.shape[0] == 1
        assert output.shape[1] > 4  # at least some tokens generated
        assert output.shape[1] <= 4 + 8 + 4  # upper bound with bonus tokens

    def test_stats_recorded(self, tiny_model):
        from open_mythos.inference.speculative import SpeculativeDepthDecoder

        tok = FakeTokenizer()
        decoder = SpeculativeDepthDecoder(
            tiny_model,
            tok,
            n_draft_loops=1,
            n_verify_loops=2,
            n_draft_tokens=2,
            device="cpu",
        )
        input_ids = torch.randint(0, 256, (1, 4))
        _ = decoder.generate(input_ids, max_new_tokens=6)
        stats = decoder.last_stats
        assert stats is not None
        assert 0.0 <= stats.acceptance_rate <= 1.0
        assert stats.draft_loops == 1
        assert stats.verify_loops == 2

    def test_greedy_deterministic(self, tiny_model):
        """Temperature=0 speculative decoding must be deterministic."""
        from open_mythos.inference.speculative import SpeculativeDepthDecoder

        tok = FakeTokenizer()
        decoder = SpeculativeDepthDecoder(
            tiny_model,
            tok,
            n_draft_loops=1,
            n_verify_loops=2,
            n_draft_tokens=2,
            device="cpu",
        )
        input_ids = torch.randint(0, 256, (1, 4))
        torch.manual_seed(0)
        out1 = decoder.generate(input_ids.clone(), max_new_tokens=4, temperature=0.0)
        torch.manual_seed(0)
        out2 = decoder.generate(input_ids.clone(), max_new_tokens=4, temperature=0.0)
        assert torch.equal(out1, out2), "Greedy speculative decoding must be deterministic"


# ─── Thesis module ────────────────────────────────────────────────────────────


class TestThesis:
    def test_thesis_string_exists(self):
        from open_mythos.core.thesis import __thesis__

        assert isinstance(__thesis__, str)
        assert len(__thesis__) > 50
        assert "recurrent" in __thesis__.lower() or "depth" in __thesis__.lower()

    def test_killer_metric_defined(self):
        from open_mythos.core.thesis import __killer_metric__

        assert __killer_metric__ == "reasoning_flop_efficiency"

    def test_moat_defined(self):
        from open_mythos.core.thesis import __moat__

        assert isinstance(__moat__, list)
        assert len(__moat__) >= 2

    def test_get_thesis_callable(self):
        from open_mythos import get_thesis

        t = get_thesis()
        assert isinstance(t, str)
        assert len(t) > 20


# ─── MoDA lm_head fix verification ───────────────────────────────────────────


class TestMoDAFix:
    def test_moda_lm_head_correct_dimensions(self):
        """Regression test: lm_head must map d_model → vocab_size (not reversed)."""
        from open_mythos.moda import MoDAConfig, MoDAModel

        cfg = MoDAConfig(
            vocab_size=100,
            d_model=64,
            n_layers=2,
            n_heads_q=4,
            n_heads_kv=2,
            head_dim=16,
            n_routed_experts=4,
            n_activated_experts=2,
            expert_hidden_dim=32,
        )
        model = MoDAModel(cfg).eval()
        x = torch.randint(0, 100, (1, 8))
        logits, _ = model(x)
        assert logits.shape == (
            1,
            8,
            100,
        ), f"lm_head output must be (B, T, vocab_size), got {logits.shape}"

    def test_moda_training_loss_finite(self):
        """Cross-entropy loss must be finite (would be NaN with wrong lm_head dims)."""
        from open_mythos.moda import MoDAConfig, MoDAModel

        cfg = MoDAConfig(
            vocab_size=100,
            d_model=64,
            n_layers=2,
            n_heads_q=4,
            n_heads_kv=2,
            head_dim=16,
            n_routed_experts=4,
            n_activated_experts=2,
            expert_hidden_dim=32,
        )
        model = MoDAModel(cfg)
        x = torch.randint(0, 100, (2, 8))
        labels = torch.randint(0, 100, (2, 8))
        _, loss = model(x, labels=labels)
        assert loss is not None
        assert torch.isfinite(loss), f"Training loss must be finite, got {loss}"
