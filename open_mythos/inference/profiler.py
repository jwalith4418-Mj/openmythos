"""
OpenMythos Inference Profiler
=============================
Measures the killer metric: Reasoning-FLOP Efficiency.

    RFE = benchmark_accuracy / mean_flops_per_token

This is the number that differentiates OpenMythos from every static model.
A static GPT-4o-class model has a fixed FLOPs/token regardless of query
difficulty. OpenMythos has variable FLOPs/token proportional to difficulty.

Usage:
    profiler = InferenceProfiler(model, tokenizer)
    report = profiler.profile_dataset(prompts, ground_truths, eval_fn)
    print(report.reasoning_flop_efficiency)  # the killer metric
    report.save_json("rfe_report.json")
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
import torch.nn as nn


@dataclass
class QueryProfile:
    """Per-query inference profile."""

    prompt: str
    n_prompt_tokens: int
    n_generated_tokens: int
    loop_counts: list[int]  # per-token loop count (elastic compute)
    total_flops: float  # estimated FLOPs for this query
    latency_ms: float
    correct: bool | None = None


@dataclass
class ProfileReport:
    """Aggregate profile over a dataset."""

    n_queries: int
    mean_loop_count: float
    p50_loop_count: float
    p95_loop_count: float
    max_loop_count: float
    mean_flops_per_token: float
    total_flops: float
    accuracy: float | None  # None if no ground truth provided
    reasoning_flop_efficiency: float | None  # accuracy / mean_flops_per_token
    mean_latency_ms: float
    p99_latency_ms: float
    query_profiles: list[QueryProfile] = field(default_factory=list)

    def save_json(self, path: str) -> None:
        d = asdict(self)
        d.pop("query_profiles")  # omit verbose per-query data
        with open(path, "w") as f:
            json.dump(d, f, indent=2)
        print(f"[Profiler] Report saved to {path}")

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  OpenMythos Inference Profile",
            "=" * 60,
            f"  Queries:              {self.n_queries}",
            f"  Mean loops/token:     {self.mean_loop_count:.2f}",
            f"  P95  loops/token:     {self.p95_loop_count:.2f}",
            f"  Max  loops/token:     {self.max_loop_count:.2f}",
            f"  Mean FLOPs/token:     {self.mean_flops_per_token:.2e}",
            f"  Mean latency:         {self.mean_latency_ms:.1f} ms",
            f"  P99  latency:         {self.p99_latency_ms:.1f} ms",
        ]
        if self.accuracy is not None:
            lines.append(f"  Accuracy:             {self.accuracy*100:.1f}%")
        if self.reasoning_flop_efficiency is not None:
            lines.append(f"  RFE (killer metric):  {self.reasoning_flop_efficiency:.4e}")
        lines.append("=" * 60)
        return "\n".join(lines)


def _estimate_flops_per_forward(model: nn.Module, seq_len: int, n_loops: int) -> float:
    """
    Estimate FLOPs for one forward pass of OpenMythos.

    For a recurrent model with n_loops iterations on the recurrent block:
        FLOPs ≈ 6 * N_params_static + n_loops * 6 * N_params_recurrent

    where N_params_static = prelude + coda params,
          N_params_recurrent = RecurrentBlock params.

    This is the per-token cost scaled by seq_len.
    """
    total = 0.0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            w = module.weight
            flops = 2.0 * w.shape[0] * w.shape[1]  # matmul: 2*m*n per token
            if "recurrent" in name:
                flops *= n_loops  # recurrent block runs n_loops times
            total += flops * seq_len
    return total


class InferenceProfiler:
    """
    Wraps OpenMythos inference and records per-query compute profiles.

    Key measurement: loop_count per token, which drives the killer metric.
    The ACT halting mechanism in RecurrentBlock produces variable loop counts.
    We instrument this to log actual compute usage.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: str = "cuda",
        max_loops: int = 64,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_loops = max_loops
        self._install_loop_hooks()

    def _install_loop_hooks(self) -> None:
        """Install forward hooks on RecurrentBlock to capture actual loop counts."""
        self._last_loop_count: list[int] = []

        def _recurrent_hook(module, input, output):
            # RecurrentBlock.forward returns h_out; we track loops via the halting
            # In forward, the loop exits early — capture via step counter injection.
            # We patch the module to store it.
            lc = getattr(module, "_last_loop_count", self.max_loops)
            self._last_loop_count.append(lc)

        for name, module in self.model.named_modules():
            if type(module).__name__ == "RecurrentBlock":
                module.register_forward_hook(_recurrent_hook)
                # Patch the module's forward to store loop count
                _patch_recurrent_block(module)

    @torch.no_grad()
    def profile_query(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        n_loops: int | None = None,
    ) -> QueryProfile:
        """Profile a single query."""
        self._last_loop_count.clear()

        ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([ids], device=self.device)
        n_prompt = len(ids)

        t0 = time.perf_counter()
        n_loops_arg = n_loops or self.max_loops
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            n_loops=n_loops_arg,
            temperature=0.0,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        n_generated = output_ids.shape[1] - n_prompt
        loop_counts = list(self._last_loop_count) or [n_loops_arg] * n_generated

        # FLOPs estimate: mean loops actually used
        mean_lc = sum(loop_counts) / max(len(loop_counts), 1)
        total_flops = _estimate_flops_per_forward(
            self.model, seq_len=n_prompt + n_generated, n_loops=int(mean_lc)
        )

        return QueryProfile(
            prompt=prompt,
            n_prompt_tokens=n_prompt,
            n_generated_tokens=n_generated,
            loop_counts=loop_counts,
            total_flops=total_flops,
            latency_ms=latency_ms,
        )

    def profile_dataset(
        self,
        prompts: list[str],
        ground_truths: list[Any] | None = None,
        eval_fn: Callable[[str, Any], bool] | None = None,
        max_new_tokens: int = 128,
        n_loops: int | None = None,
    ) -> ProfileReport:
        """Profile a list of prompts and compute aggregate metrics."""
        profiles: list[QueryProfile] = []
        correct_count = 0

        for i, prompt in enumerate(prompts):
            profile = self.profile_query(prompt, max_new_tokens, n_loops)

            if ground_truths is not None and eval_fn is not None:
                ids = self.tokenizer.encode(prompt)
                input_ids = torch.tensor([ids], device=self.device)
                with torch.no_grad():
                    output_ids = self.model.generate(
                        input_ids,
                        max_new_tokens=max_new_tokens,
                        n_loops=n_loops or self.max_loops,
                        temperature=0.0,
                    )
                generated = self.tokenizer.decode(output_ids[0, len(ids) :].tolist())
                profile.correct = eval_fn(generated, ground_truths[i])
                if profile.correct:
                    correct_count += 1

            profiles.append(profile)
            if (i + 1) % 50 == 0:
                print(f"[Profiler] {i+1}/{len(prompts)} queries profiled")

        # Aggregate
        all_loops = [lc for p in profiles for lc in (p.loop_counts or [self.max_loops])]
        all_loops_sorted = sorted(all_loops)
        n = len(all_loops_sorted)
        latencies_sorted = sorted(p.latency_ms for p in profiles)

        mean_loops = sum(all_loops) / max(n, 1)
        total_flops = sum(p.total_flops for p in profiles)
        mean_flops_per_token = total_flops / max(sum(p.n_generated_tokens for p in profiles), 1)

        accuracy = correct_count / max(len(profiles), 1) if ground_truths is not None else None
        rfe = accuracy / max(mean_flops_per_token, 1e-30) if accuracy is not None else None

        report = ProfileReport(
            n_queries=len(profiles),
            mean_loop_count=mean_loops,
            p50_loop_count=all_loops_sorted[n // 2] if n > 0 else 0,
            p95_loop_count=all_loops_sorted[int(n * 0.95)] if n > 0 else 0,
            max_loop_count=max(all_loops_sorted, default=0),
            mean_flops_per_token=mean_flops_per_token,
            total_flops=total_flops,
            accuracy=accuracy,
            reasoning_flop_efficiency=rfe,
            mean_latency_ms=sum(p.latency_ms for p in profiles) / max(len(profiles), 1),
            p99_latency_ms=(
                latencies_sorted[int(len(latencies_sorted) * 0.99)] if latencies_sorted else 0
            ),
            query_profiles=profiles,
        )

        print(report.summary())
        return report


def _patch_recurrent_block(module: nn.Module) -> None:
    """
    Monkey-patch RecurrentBlock.forward to store the actual loop count used.
    This lets the profiler measure actual elastic compute without modifying
    the core model code.
    """

    def patched_forward(h, e, freqs_cis, mask=None, n_loops=None, kv_cache=None):
        # We intercept the ACT loop by counting iterations before halting.
        # The original forward already breaks early on halted.all().
        # We run it and record the count via the module attribute.
        import torch as _torch

        cfg = module.cfg
        n_loops = n_loops or cfg.max_loop_iters
        B, T, D = h.shape
        halted = _torch.zeros(B, T, device=h.device, dtype=_torch.bool)
        cumulative_p = _torch.zeros(B, T, device=h.device)
        h_out = _torch.zeros_like(h)
        actual_loops = 0

        for t in range(n_loops):
            actual_loops = t + 1
            h = module._loop_step(h, e, freqs_cis, mask, kv_cache, t)
            p = module.act(h)
            still_running = ~halted
            remainder = (1.0 - cumulative_p).clamp(min=0)
            weight = _torch.where(cumulative_p + p >= cfg.act_threshold, remainder, p)
            weight = weight * still_running.float()
            h_out = h_out + weight.unsqueeze(-1) * h
            cumulative_p = cumulative_p + p * still_running.float()
            halted = halted | (cumulative_p >= cfg.act_threshold)
            if halted.all():
                break

        module._last_loop_count = actual_loops
        return h_out

    module.forward = patched_forward
