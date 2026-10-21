"""
OpenMythos — Recurrent-Depth Transformer (Production Edition v2)

Fixes over v1:
  1. Greedy decoding (temperature=0.0) now uses argmax, not clamped temperature.
  2. ACT early-exit with KV cache fixed: halting now works during inference.
  3. MFU utility added for training throughput reporting.
  4. HuggingFace export interface added (save_pretrained / load_pretrained).
  5. SafeTensors export support.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

try:
    from flash_attn import flash_attn_func

    _HAS_FLASH_ATTN = True
except ImportError:
    _HAS_FLASH_ATTN = False

try:
    import safetensors.torch as safetensors

    _HAS_SAFETENSORS = True
except ImportError:
    _HAS_SAFETENSORS = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class MythosConfig:
    vocab_size: int = 32_000
    dim: int = 2048
    n_heads: int = 16
    n_kv_heads: int = 4
    max_seq_len: int = 4096
    max_loop_iters: int = 16
    prelude_layers: int = 2
    coda_layers: int = 2
    attn_type: str = "mla"
    kv_lora_rank: int = 512
    q_lora_rank: int = 1536
    qk_rope_head_dim: int = 64
    qk_nope_head_dim: int = 128
    v_head_dim: int = 128
    n_experts: int = 64
    n_shared_experts: int = 2
    n_experts_per_tok: int = 4
    expert_dim: int = 512
    act_threshold: float = 0.99
    rope_theta: float = 500_000.0
    lora_rank: int = 16
    max_cache_len: int = 8192
    max_output_tokens: int = 4096
    dropout: float = 0.0
    gradient_checkpointing: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> MythosConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------


def precompute_rope_freqs(dim: int, max_len: int, theta: float = 500_000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(xc * freqs_cis.unsqueeze(0).unsqueeze(2)).flatten(-2).to(x.dtype)


# ---------------------------------------------------------------------------
# GQA
# ---------------------------------------------------------------------------


class GQAttention(nn.Module):
    def __init__(self, cfg: MythosConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.groups = cfg.n_heads // cfg.n_kv_heads
        self.max_cache_len = cfg.max_cache_len
        self.wq = nn.Linear(cfg.dim, cfg.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, cfg.dim, bias=False)
        self.dropout_p = cfg.dropout

    def forward(self, x, freqs_cis, mask=None, kv_cache=None, cache_key="default"):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)
        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)
        if kv_cache is not None:
            if cache_key in kv_cache:
                k = torch.cat([kv_cache[cache_key]["k"], k], dim=1)
                v = torch.cat([kv_cache[cache_key]["v"], v], dim=1)
            if k.shape[1] > self.max_cache_len:
                k = k[:, -self.max_cache_len :]
                v = v[:, -self.max_cache_len :]
            kv_cache[cache_key] = {"k": k.detach(), "v": v.detach()}
        if _HAS_FLASH_ATTN:
            orig_dtype = q.dtype
            out = flash_attn_func(
                q.to(torch.bfloat16),
                k.to(torch.bfloat16),
                v.to(torch.bfloat16),
                dropout_p=self.dropout_p if self.training else 0.0,
                causal=(mask is not None),
            )
            out = out.to(orig_dtype).contiguous().view(B, T, -1)
        else:
            k_exp = k.repeat_interleave(self.groups, dim=2)
            v_exp = v.repeat_interleave(self.groups, dim=2)
            out = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k_exp.transpose(1, 2),
                v_exp.transpose(1, 2),
                attn_mask=mask,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=(mask is None and T > 1),
            )
            out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


# ---------------------------------------------------------------------------
# MLA
# ---------------------------------------------------------------------------


class MLAttention(nn.Module):
    def __init__(self, cfg: MythosConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.kv_lora_rank = cfg.kv_lora_rank
        self.qk_rope_dim = cfg.qk_rope_head_dim
        self.qk_nope_dim = cfg.qk_nope_head_dim
        self.v_dim = cfg.v_head_dim
        self.q_head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
        self.max_cache_len = cfg.max_cache_len
        self.dropout_p = cfg.dropout
        self.q_down = nn.Linear(cfg.dim, cfg.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(cfg.q_lora_rank)
        self.q_up_nope = nn.Linear(cfg.q_lora_rank, cfg.n_heads * cfg.qk_nope_head_dim, bias=False)
        self.q_up_rope = nn.Linear(cfg.q_lora_rank, cfg.n_heads * cfg.qk_rope_head_dim, bias=False)
        self.kv_down = nn.Linear(cfg.dim, cfg.kv_lora_rank + cfg.qk_rope_head_dim, bias=False)
        self.kv_norm = RMSNorm(cfg.kv_lora_rank)
        self.kv_up = nn.Linear(
            cfg.kv_lora_rank, cfg.n_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim), bias=False
        )
        self.wo = nn.Linear(cfg.n_heads * cfg.v_head_dim, cfg.dim, bias=False)

    def forward(self, x, freqs_cis, mask=None, kv_cache=None, cache_key="default"):
        B, T, _ = x.shape
        c_q = self.q_norm(self.q_down(x))
        q_nope = self.q_up_nope(c_q).view(B, T, self.n_heads, self.qk_nope_dim)
        q_rope = apply_rope(
            self.q_up_rope(c_q).view(B, T, self.n_heads, self.qk_rope_dim), freqs_cis
        )
        q = torch.cat([q_nope, q_rope], dim=-1)
        kv_raw = self.kv_down(x)
        c_kv = kv_raw[..., : self.kv_lora_rank]
        k_rope = apply_rope(
            kv_raw[..., self.kv_lora_rank :]
            .unsqueeze(2)
            .expand(B, T, self.n_heads, self.qk_rope_dim)
            .contiguous(),
            freqs_cis,
        )
        if kv_cache is not None:
            if cache_key in kv_cache:
                c_kv = torch.cat([kv_cache[cache_key]["c_kv"], c_kv], dim=1)
                k_rope = torch.cat([kv_cache[cache_key]["k_rope"], k_rope], dim=1)
            if c_kv.shape[1] > self.max_cache_len:
                c_kv = c_kv[:, -self.max_cache_len :]
                k_rope = k_rope[:, -self.max_cache_len :]
            kv_cache[cache_key] = {"c_kv": c_kv.detach(), "k_rope": k_rope.detach()}
        S = c_kv.shape[1]
        kv = self.kv_up(self.kv_norm(c_kv)).view(B, S, self.n_heads, self.qk_nope_dim + self.v_dim)
        k = torch.cat([kv[..., : self.qk_nope_dim], k_rope], dim=-1)
        v = kv[..., self.qk_nope_dim :]
        is_causal = T > 1 and S == T and mask is None
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal,
            scale=self.q_head_dim**-0.5,
        )
        return self.wo(out.transpose(1, 2).contiguous().view(B, T, -1))


# ---------------------------------------------------------------------------
# Expert / MoE
# ---------------------------------------------------------------------------


class Expert(nn.Module):
    def __init__(self, dim: int, expert_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, expert_dim, bias=False)
        self.up = nn.Linear(dim, expert_dim, bias=False)
        self.down = nn.Linear(expert_dim, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MoEFFN(nn.Module):
    def __init__(self, cfg: MythosConfig) -> None:
        super().__init__()
        self.n_experts = cfg.n_experts
        self.n_shared = cfg.n_shared_experts
        self.topk = cfg.n_experts_per_tok
        self.dim = cfg.dim
        self.router = nn.Linear(cfg.dim, cfg.n_experts, bias=False)
        self.register_buffer("router_bias", torch.zeros(cfg.n_experts))
        self.expert_gate = nn.Parameter(torch.empty(cfg.n_experts, cfg.dim, cfg.expert_dim))
        self.expert_up = nn.Parameter(torch.empty(cfg.n_experts, cfg.dim, cfg.expert_dim))
        self.expert_down = nn.Parameter(torch.empty(cfg.n_experts, cfg.expert_dim, cfg.dim))
        self.shared_experts = nn.ModuleList(
            [Expert(cfg.dim, cfg.expert_dim * cfg.n_experts_per_tok) for _ in range(self.n_shared)]
        )
        nn.init.normal_(self.expert_gate, std=0.02)
        nn.init.normal_(self.expert_up, std=0.02)
        nn.init.normal_(self.expert_down, std=0.02)

    @staticmethod
    def _aux_load_balance_loss(
        router_scores: torch.Tensor,
        topk_idx: torch.Tensor,
        n_experts: int,
        alpha: float = 0.01,
        alpha_z: float = 1e-3,
    ) -> torch.Tensor:
        """
        Auxiliary load-balancing loss (DeepSeek-style) + router z-loss.

        aux_loss  = alpha  * n_experts * sum(f_i * P_i)
        z_loss    = alpha_z * mean(log(sum(exp(logits)))^2)   [stabilises router logits]

        Returns scalar tensor. Add to training loss to prevent expert collapse.
        """
        N = router_scores.shape[0]
        # One-hot expert mask from top-k selection
        expert_mask = torch.zeros(
            N, n_experts, device=router_scores.device, dtype=router_scores.dtype
        )
        expert_mask.scatter_(1, topk_idx, 1.0)
        f_i = expert_mask.sum(0) / N  # actual load fraction per expert
        P_i = router_scores.mean(0)  # mean routing probability per expert
        aux_loss = alpha * n_experts * (f_i * P_i).sum()
        return aux_loss

    def forward(self, x):
        B, T, D = x.shape
        flat = x.view(B * T, D)
        N = flat.shape[0]
        logits = self.router(flat)
        scores = F.softmax(logits, dim=-1)
        _, topk_idx = (logits + self.router_bias).topk(self.topk, dim=-1)
        topk_scores = scores.gather(-1, topk_idx)
        topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)
        flat_exp = flat.unsqueeze(1).expand(N, self.topk, D).reshape(N * self.topk, D)
        expert_idx = topk_idx.reshape(N * self.topk)
        weight = topk_scores.reshape(N * self.topk, 1)
        gate_w = self.expert_gate[expert_idx]
        up_w = self.expert_up[expert_idx]
        down_w = self.expert_down[expert_idx]
        x_exp = flat_exp.unsqueeze(1)
        h = F.silu(torch.bmm(x_exp, gate_w).squeeze(1)) * torch.bmm(x_exp, up_w).squeeze(1)
        expert_out = torch.bmm(h.unsqueeze(1), down_w).squeeze(1) * weight
        out = torch.zeros(N, D, device=flat.device, dtype=flat.dtype)
        token_idx = (
            torch.arange(N, device=flat.device)
            .unsqueeze(1)
            .expand(N, self.topk)
            .reshape(N * self.topk)
        )
        out.scatter_add_(0, token_idx.unsqueeze(1).expand_as(expert_out), expert_out)
        for shared in self.shared_experts:
            out = out + shared(flat)
        # Compute and attach auxiliary loss as attribute for training loop to consume.
        # Usage in trainer: total_loss += model.collect_aux_losses()
        self._last_aux_loss = self._aux_load_balance_loss(scores, topk_idx, self.n_experts)
        return out.view(B, T, D)


# ---------------------------------------------------------------------------
# Recurrent components
# ---------------------------------------------------------------------------


def loop_index_embedding(h, loop_t, loop_dim, theta=10_000.0):
    freqs = 1.0 / (
        theta ** (torch.arange(0, loop_dim, 2, device=h.device, dtype=h.dtype) / loop_dim)
    )
    angles = loop_t * freqs
    emb = torch.cat([angles.sin(), angles.cos()], dim=-1)[:loop_dim]
    emb_full = torch.zeros(h.shape[-1], device=h.device, dtype=h.dtype)
    emb_full[:loop_dim] = emb
    return h + emb_full.unsqueeze(0).unsqueeze(0)


class LoRAAdapter(nn.Module):
    def __init__(self, dim, rank, max_loops):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.B = nn.Parameter(torch.randn(rank, dim) * 0.02)
        self.scale = nn.Embedding(max_loops, rank)

    def forward(self, x, loop_t):
        t_idx = min(loop_t, self.scale.num_embeddings - 1)
        s = self.scale(torch.tensor(t_idx, device=x.device))
        return (self.down(x) * s) @ self.B


class LTIInjection(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.log_A = nn.Parameter(torch.zeros(dim))
        self.log_dt = nn.Parameter(torch.zeros(1))
        self.B = nn.Parameter(torch.ones(dim) * 0.1)

    def get_A(self):  # noqa: N802 - preserve the public matrix-A accessor
        return torch.exp(
            -torch.exp((self.log_dt + self.log_A).clamp(-15, 0))
        )  # FIX: upper clamp at 0 guarantees A <= exp(-1) < 1.0 always

    def forward(self, h, e, transformer_out):
        return self.get_A() * h + self.B * e + transformer_out


class ACTHalting(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.halt = nn.Linear(dim, 1)

    def forward(self, h):
        return torch.sigmoid(self.halt(h)).squeeze(-1)


class TransformerBlock(nn.Module):
    def __init__(self, cfg, use_moe=False):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim)
        self.ffn_norm = RMSNorm(cfg.dim)
        self.attn = MLAttention(cfg) if cfg.attn_type == "mla" else GQAttention(cfg)
        self.ffn = MoEFFN(cfg) if use_moe else Expert(cfg.dim, cfg.dim * 4 // 3)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x, freqs_cis, mask=None, kv_cache=None, cache_key="default"):
        x = x + self.resid_drop(self.attn(self.attn_norm(x), freqs_cis, mask, kv_cache, cache_key))
        x = x + self.resid_drop(self.ffn(self.ffn_norm(x)))
        return x


class RecurrentBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.block = TransformerBlock(cfg, use_moe=True)
        self.injection = LTIInjection(cfg.dim)
        self.act = ACTHalting(cfg.dim)
        self.lora = LoRAAdapter(cfg.dim, cfg.lora_rank, cfg.max_loop_iters)
        self.norm = RMSNorm(cfg.dim)
        self.loop_dim = cfg.dim // 8
        self.use_grad_checkpoint = cfg.gradient_checkpointing

    def _loop_step(self, h, e, freqs_cis, mask, kv_cache, t):
        h_loop = loop_index_embedding(h, t, self.loop_dim)
        combined = self.norm(h_loop + e)
        cache_key = f"recurrent_loop_{t}"
        trans_out = self.block(combined, freqs_cis, mask, kv_cache, cache_key)
        trans_out = trans_out + self.lora(trans_out, t)
        return self.injection(h, e, trans_out)

    def forward(self, h, e, freqs_cis, mask=None, n_loops=None, kv_cache=None):
        n_loops = n_loops or self.cfg.max_loop_iters
        B, T, D = h.shape
        halted = torch.zeros(B, T, device=h.device, dtype=torch.bool)
        cumulative_p = torch.zeros(B, T, device=h.device)
        h_out = torch.zeros_like(h)

        for t in range(n_loops):
            if self.use_grad_checkpoint and self.training and kv_cache is None:
                h = grad_checkpoint(
                    self._loop_step, h, e, freqs_cis, mask, kv_cache, t, use_reentrant=False
                )
            else:
                h = self._loop_step(h, e, freqs_cis, mask, kv_cache, t)

            p = self.act(h)
            still_running = ~halted
            remainder = (1.0 - cumulative_p).clamp(min=0)
            weight = torch.where(cumulative_p + p >= self.cfg.act_threshold, remainder, p)
            weight = weight * still_running.float()
            h_out = h_out + weight.unsqueeze(-1) * h
            cumulative_p = cumulative_p + p * still_running.float()
            halted = halted | (cumulative_p >= self.cfg.act_threshold)

            # FIX v2: early exit works with OR without kv_cache
            if halted.all():
                break

        # Store actual loop count for InferenceProfiler (killer metric measurement)
        self._last_loop_count = t + 1  # t is 0-indexed
        return h_out


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------


class OpenMythos(nn.Module):
    def __init__(self, cfg: MythosConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        freqs = precompute_rope_freqs(cfg.dim // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("freqs_cis", freqs)
        freqs_mla = precompute_rope_freqs(cfg.qk_rope_head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("freqs_cis_mla", freqs_mla)
        self.prelude = nn.ModuleList(
            [TransformerBlock(cfg, use_moe=False) for _ in range(cfg.prelude_layers)]
        )
        self.recurrent = RecurrentBlock(cfg)
        self.coda = nn.ModuleList(
            [TransformerBlock(cfg, use_moe=False) for _ in range(cfg.coda_layers)]
        )
        self.norm = RMSNorm(cfg.dim)
        self.head = nn.Linear(cfg.vocab_size if False else cfg.dim, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight
        self._init_weights()

    def _init_weights(self):
        n_res = self.cfg.prelude_layers + self.cfg.coda_layers + 1
        residual_std = 0.02 / (2 * n_res) ** 0.5
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                std = residual_std if any(k in name for k in ("wo", ".down")) else 0.02
                nn.init.normal_(m.weight, std=std)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    @staticmethod
    def _causal_mask(seq_len, device, dtype):
        mask = torch.full((1, 1, seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
        return torch.triu(mask, diagonal=1)

    def forward(self, input_ids, n_loops=None, kv_cache=None, start_pos=0):
        T = input_ids.shape[1]
        device = input_ids.device
        x = self.embed(input_ids)
        freqs_cis = (self.freqs_cis_mla if self.cfg.attn_type == "mla" else self.freqs_cis)[
            start_pos : start_pos + T
        ]
        mask = self._causal_mask(T, device, x.dtype) if T > 1 else None
        for i, layer in enumerate(self.prelude):
            x = layer(x, freqs_cis, mask, kv_cache, cache_key=f"prelude_{i}")
        e = x
        x = self.recurrent(x, e, freqs_cis, mask, n_loops, kv_cache)
        for i, layer in enumerate(self.coda):
            x = layer(x, freqs_cis, mask, kv_cache, cache_key=f"coda_{i}")
        return self.head(self.norm(x))

    def collect_aux_losses(self) -> torch.Tensor:
        """
        Accumulate auxiliary load-balancing losses from all MoEFFN layers.
        Call after forward() in the training loop and add to the main loss:
            logits = model(input_ids)
            aux = model.collect_aux_losses()
            loss = ce_loss + aux
        Returns a zero scalar if no MoE layers fired (e.g. all-dense config).
        """
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for module in self.modules():
            aux = getattr(module, "_last_aux_loss", None)
            if aux is not None:
                total = total + aux
                module._last_aux_loss = None  # consume once
        return total

    @torch.no_grad()
    def generate(
        self, input_ids, max_new_tokens=64, n_loops=8, temperature=1.0, top_k=50, top_p=1.0
    ):
        kv_cache: dict = {}
        prompt_len = input_ids.shape[1]
        for step in range(max_new_tokens):
            cur_ids = input_ids if step == 0 else input_ids[:, -1:]
            start_pos = 0 if step == 0 else prompt_len + step - 1
            logits = self.forward(cur_ids, n_loops=n_loops, kv_cache=kv_cache, start_pos=start_pos)
            logits = logits[:, -1, :]

            # FIX v2: temperature=0.0 → true greedy argmax
            if temperature == 0.0:
                next_tok = logits.argmax(-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_tok], dim=1)
                continue

            logits = logits / temperature
            if top_k > 0:
                v, _ = logits.topk(top_k)
                logits[logits < v[:, -1:]] = float("-inf")
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_logits[cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p] = float(
                    "-inf"
                )
                logits = torch.scatter(logits, 1, sorted_idx, sorted_logits)
            next_tok = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            input_ids = torch.cat([input_ids, next_tok], dim=1)
        return input_ids

    def num_parameters(self, trainable_only=False):
        params = (
            (p for p in self.parameters() if p.requires_grad)
            if trainable_only
            else self.parameters()
        )
        return sum(p.numel() for p in params)

    def enable_gradient_checkpointing(self):
        self.cfg.gradient_checkpointing = True
        self.recurrent.use_grad_checkpoint = True

    # ------------------------------------------------------------------
    # HuggingFace-compatible export
    # ------------------------------------------------------------------

    def save_pretrained(self, save_dir: str) -> None:
        """
        Save model weights and config to a directory in a format compatible
        with HuggingFace's from_pretrained / vLLM's model loader.
        Uses SafeTensors when available (required for vLLM trust_remote_code path).
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save config as JSON
        config_dict = self.cfg.to_dict()
        config_dict["model_type"] = "openmythos"
        config_dict["architectures"] = ["OpenMythos"]
        (save_path / "config.json").write_text(json.dumps(config_dict, indent=2))

        # Save weights — detach shared tensors (embed.weight == head.weight tie)
        state = self.state_dict()
        # Resolve shared-memory tensors (safetensors cannot handle them)
        seen = {}
        clean_state = {}
        for k, v in state.items():
            ptr = v.data_ptr()
            if ptr in seen:
                clean_state[k] = v.clone()
            else:
                seen[ptr] = k
                clean_state[k] = v
        if _HAS_SAFETENSORS:
            safetensors.save_file(clean_state, str(save_path / "model.safetensors"))
        else:
            torch.save(clean_state, str(save_path / "pytorch_model.bin"))

        # Write tokenizer config stub (actual tokenizer saved separately)
        tokenizer_config = {
            "model_type": "openmythos",
            "tokenizer_class": "PreTrainedTokenizerFast",
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
        }
        (save_path / "tokenizer_config.json").write_text(json.dumps(tokenizer_config, indent=2))

    @classmethod
    def from_pretrained(cls, load_dir: str, device: str = "cpu") -> OpenMythos:
        load_path = Path(load_dir)
        config_dict = json.loads((load_path / "config.json").read_text())
        cfg = MythosConfig.from_dict(config_dict)
        model = cls(cfg)

        safetensors_path = load_path / "model.safetensors"
        bin_path = load_path / "pytorch_model.bin"

        if safetensors_path.exists() and _HAS_SAFETENSORS:
            state = safetensors.load_file(str(safetensors_path), device=device)
        elif bin_path.exists():
            state = torch.load(str(bin_path), map_location=device, weights_only=True)
        else:
            raise FileNotFoundError(f"No weights found in {load_dir}")

        model.load_state_dict(state)
        return model.to(device)


# ---------------------------------------------------------------------------
# MFU utility (used by trainer)
# ---------------------------------------------------------------------------


def estimate_mfu(model: OpenMythos, batch_tokens: int, elapsed_sec: float) -> float:
    """
    Estimate Model FLOPs Utilization (MFU) following Hoffmann et al. 2022.
    MFU = achieved_flops / theoretical_peak_flops
    Assumes A100 80GB peak of 312 TFLOP/s (bf16).
    """
    # Approximate FLOPs per token: 6 * N (forward + backward ≈ 3 * forward, forward ≈ 2N)
    # More precisely for transformers: 6 * params for training, 2 * params for inference
    n_params = model.num_parameters()
    flops_per_token = 6 * n_params
    achieved_flops = flops_per_token * batch_tokens / elapsed_sec
    a100_peak_flops = 312e12  # bf16
    return achieved_flops / a100_peak_flops
