"""
Alignment Training for OpenMythos.

Implements:
  - DPO (Direct Preference Optimization) — Rafailov et al., 2023
  - GRPO (Group Relative Policy Optimization) — DeepSeek-R1, 2025
  - SFT (Supervised Fine-Tuning) with packing

DPO is the highest-ROI alignment method:
  - No reward model required
  - Stable training (no RL instability)
  - 2× more parameter-efficient than PPO (no value head)
  - Proven at scale: Llama-2-Chat, Zephyr, Mistral-Instruct all use DPO

GRPO is used for reasoning/math tasks:
  - Group multiple rollouts, use relative reward (no baseline estimation)
  - Used in DeepSeek-R1 to achieve o1-level reasoning
  - Lower variance than PPO with group normalization

Usage (DPO):
    trainer = DPOTrainer(model, ref_model, config)
    trainer.train(preference_dataset)

Usage (GRPO):
    trainer = GRPOTrainer(model, reward_fn, config)
    trainer.train(prompt_dataset)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DPOConfig:
    """DPO training configuration."""

    beta: float = 0.1  # KL regularization strength
    label_smoothing: float = 0.0  # smoothing on winning/losing labels
    loss_type: str = "sigmoid"  # "sigmoid" | "ipo" | "kto_pair"
    reference_free: bool = False  # skip ref model (SLiC-style)
    lr: float = 5e-7  # much lower than pretraining
    weight_decay: float = 0.0
    max_seq_len: int = 2048
    micro_batch_size: int = 2
    grad_accum_steps: int = 4
    max_steps: int = 10_000
    warmup_steps: int = 100
    eval_every: int = 500
    ckpt_dir: str = "checkpoints/dpo"


@dataclass
class GRPOConfig:
    """GRPO training configuration (DeepSeek-R1 style)."""

    group_size: int = 8  # G: rollouts per prompt
    epsilon: float = 0.2  # PPO clip ratio
    beta: float = 0.001  # KL penalty weight
    temperature: float = 1.0  # sampling temperature
    max_new_tokens: int = 512
    lr: float = 1e-6
    micro_batch_size: int = 1
    grad_accum_steps: int = 8
    max_steps: int = 5_000
    warmup_steps: int = 50


# ---------------------------------------------------------------------------
# DPO Trainer
# ---------------------------------------------------------------------------


class DPOTrainer:
    """
    Direct Preference Optimization trainer.

    Reference: "Direct Preference Optimization: Your Language Model is
    Secretly a Reward Model" (Rafailov et al., 2023).

    The DPO loss is:
        L_DPO = -E[ log σ( β · (log π_θ(yw|x)/π_ref(yw|x)
                              - log π_θ(yl|x)/π_ref(yl|x)) ) ]

    where yw = chosen response, yl = rejected response.

    This is equivalent to policy gradient with implicit reward
        r(x, y) = β · log(π_θ(y|x) / π_ref(y|x))
    but avoids the instability of RL by solving the constrained
    optimization in closed form.
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module | None,
        cfg: DPOConfig,
        vocab_size: int,
    ) -> None:
        self.model = model
        self.ref_model = ref_model
        self.cfg = cfg
        self.vocab_size = vocab_size

        if ref_model is not None:
            ref_model.eval()
            for p in ref_model.parameters():
                p.requires_grad_(False)

        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

    def get_log_probs(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-token log probabilities for the response portion.

        Args:
            input_ids: (B, T) full sequence (prompt + response)
            labels: (B, T) with -100 for prompt tokens (ignored in loss)

        Returns:
            (B,) sum of log probs over non-ignored response tokens
        """
        with torch.no_grad() if model is self.ref_model else torch.enable_grad():
            logits = model(input_ids)  # (B, T, V)

        # Shift: predict token t+1 from hidden state at t
        shift_logits = logits[:, :-1, :]  # (B, T-1, V)
        shift_labels = labels[:, 1:]  # (B, T-1)

        log_probs = F.log_softmax(shift_logits, dim=-1)

        # Gather log prob of the correct token at each position
        # Shape: (B, T-1) — only response tokens (where label != -100)
        per_token = log_probs.gather(-1, shift_labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)

        # Mask out prompt tokens (-100) and pad
        mask = (shift_labels != -100).float()
        return (per_token * mask).sum(dim=-1)  # (B,) summed over response

    def dpo_loss(
        self,
        chosen_ids: torch.Tensor,
        chosen_labels: torch.Tensor,
        rejected_ids: torch.Tensor,
        rejected_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute DPO loss for a batch of preference pairs.

        Returns:
            loss: scalar DPO loss
            metrics: dict with reward_chosen, reward_rejected, reward_margin, accuracy
        """
        # Policy log probs
        pi_logprob_w = self.get_log_probs(self.model, chosen_ids, chosen_labels)
        pi_logprob_l = self.get_log_probs(self.model, rejected_ids, rejected_labels)

        # Reference log probs
        if self.ref_model is not None and not self.cfg.reference_free:
            with torch.no_grad():
                ref_logprob_w = self.get_log_probs(self.ref_model, chosen_ids, chosen_labels)
                ref_logprob_l = self.get_log_probs(self.ref_model, rejected_ids, rejected_labels)
        else:
            ref_logprob_w = torch.zeros_like(pi_logprob_w)
            ref_logprob_l = torch.zeros_like(pi_logprob_l)

        # Log ratios: log(π/π_ref)
        log_ratio_w = pi_logprob_w - ref_logprob_w
        log_ratio_l = pi_logprob_l - ref_logprob_l

        if self.cfg.loss_type == "sigmoid":
            # Standard DPO loss
            losses = -F.logsigmoid(self.cfg.beta * (log_ratio_w - log_ratio_l))
            if self.cfg.label_smoothing > 0:
                losses = (1 - self.cfg.label_smoothing) * losses + self.cfg.label_smoothing * (
                    -F.logsigmoid(-(self.cfg.beta * (log_ratio_w - log_ratio_l)))
                )

        elif self.cfg.loss_type == "ipo":
            # IPO loss (Azar et al., 2024): less overoptimization at high β
            losses = (log_ratio_w - log_ratio_l - 1 / (2 * self.cfg.beta)) ** 2

        elif self.cfg.loss_type == "kto_pair":
            # KTO-Pair (Ethayarajh et al., 2024): uses KL-divergence directly
            kl = (log_ratio_w - log_ratio_l).mean().detach()
            losses = -F.logsigmoid(self.cfg.beta * (log_ratio_w - kl)) - F.logsigmoid(
                -(self.cfg.beta * (log_ratio_l - kl))
            )
        else:
            raise ValueError(f"Unknown loss_type: {self.cfg.loss_type}")

        loss = losses.mean()

        # Metrics for monitoring
        with torch.no_grad():
            rewards_w = self.cfg.beta * log_ratio_w.detach()
            rewards_l = self.cfg.beta * log_ratio_l.detach()
            metrics = {
                "reward_chosen": rewards_w.mean().item(),
                "reward_rejected": rewards_l.mean().item(),
                "reward_margin": (rewards_w - rewards_l).mean().item(),
                "accuracy": (rewards_w > rewards_l).float().mean().item(),
            }

        return loss, metrics

    def train_step(
        self,
        chosen_ids: torch.Tensor,
        chosen_labels: torch.Tensor,
        rejected_ids: torch.Tensor,
        rejected_labels: torch.Tensor,
    ) -> dict[str, float]:
        """Single DPO training step."""
        self.model.train()
        loss, metrics = self.dpo_loss(chosen_ids, chosen_labels, rejected_ids, rejected_labels)
        loss.backward()
        metrics["loss"] = loss.item()
        return metrics

    def train(self, dataloader, eval_dataloader=None) -> None:
        """Full DPO training loop."""
        self.model.train()
        step = 0
        self.optimizer.zero_grad()
        micro = 0

        for batch in dataloader:
            if step >= self.cfg.max_steps:
                break

            chosen_ids = batch["chosen_input_ids"].cuda()
            chosen_labels = batch["chosen_labels"].cuda()
            rejected_ids = batch["rejected_input_ids"].cuda()
            rejected_labels = batch["rejected_labels"].cuda()

            metrics = self.train_step(chosen_ids, chosen_labels, rejected_ids, rejected_labels)
            micro += 1

            if micro == self.cfg.grad_accum_steps:
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                step += 1
                micro = 0

                if step % 10 == 0:
                    print(
                        f"DPO step {step}/{self.cfg.max_steps} | "
                        f"loss {metrics['loss']:.4f} | "
                        f"margin {metrics['reward_margin']:.4f} | "
                        f"acc {metrics['accuracy']:.3f}"
                    )


# ---------------------------------------------------------------------------
# GRPO Trainer (Group Relative Policy Optimization)
# ---------------------------------------------------------------------------


class GRPOTrainer:
    """
    Group Relative Policy Optimization trainer.

    Reference: DeepSeek-R1 (2025). Used for reasoning/math tasks where
    ground-truth verifiable rewards are available.

    GRPO algorithm:
        1. For each prompt, generate G completions with the policy π_θ.
        2. Compute reward r_i for each completion (e.g., math answer match).
        3. Normalize rewards within the group: Â_i = (r_i - mean(r)) / std(r).
        4. Clip importance ratio like PPO: ρ = π_θ(y|x) / π_old(y|x).
        5. GRPO loss: -E[ min(ρ · Â, clip(ρ, 1-ε, 1+ε) · Â) ] + β·KL(π||π_ref).

    Key insight: group normalization eliminates the need for a learned value
    function (critic), making GRPO simpler and more stable than PPO.
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        reward_fn: Callable[[list[str], str], list[float]],
        cfg: GRPOConfig,
        tokenizer,
    ) -> None:
        self.model = model
        self.ref_model = ref_model
        self.reward_fn = reward_fn  # fn(completions: List[str], prompt: str) -> List[float]
        self.cfg = cfg
        self.tokenizer = tokenizer

        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)

        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.lr,
        )

    def _get_token_log_probs(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,  # (B, T)
        response_mask: torch.Tensor,  # (B, T) 1 for response tokens, 0 for prompt
    ) -> torch.Tensor:
        """Compute sum of log probs over response tokens."""
        logits = model(input_ids)
        shift_logits = logits[:, :-1, :]
        shift_ids = input_ids[:, 1:]
        shift_mask = response_mask[:, 1:]

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(-1, shift_ids.unsqueeze(-1)).squeeze(-1)
        return (token_log_probs * shift_mask).sum(dim=-1)  # (B,)

    def grpo_step(
        self,
        prompt_ids: torch.Tensor,  # (1, T_prompt)
    ) -> dict[str, float]:
        """
        Single GRPO update step for one prompt.

        1. Sample G completions.
        2. Score with reward function.
        3. Compute advantage via group normalization.
        4. Compute clipped policy gradient loss + KL penalty.
        """
        G = self.cfg.group_size
        device = prompt_ids.device

        # 1. Generate G completions with current policy
        self.model.eval()
        with torch.no_grad():
            # Expand prompt G times for batch generation
            prompt_expanded = prompt_ids.expand(G, -1)
            completions = self.model.generate(
                prompt_expanded,
                max_new_tokens=self.cfg.max_new_tokens,
                temperature=self.cfg.temperature,
                top_k=50,
            )  # (G, T_prompt + T_response)

            # FIX: snapshot π_old logprobs immediately after rollout, while model is still in eval().
            # This is the true "old policy" for importance sampling — NOT ref_model.
            # ref_model is used separately for KL regularization only.
            pi_old_logprobs = self._get_token_log_probs(
                self.model, completions, torch.ones_like(completions)
            )  # (G,) — frozen snapshot of π_θ at rollout time

        prompt_len = prompt_ids.shape[1]
        response_ids = completions[:, prompt_len:]  # (G, T_response)
        response_mask = torch.ones_like(completions)
        response_mask[:, :prompt_len] = 0

        # 2. Score completions
        response_texts = [self.tokenizer.decode(r.tolist()) for r in response_ids]
        prompt_text = self.tokenizer.decode(prompt_ids[0].tolist())
        rewards = self.reward_fn(response_texts, prompt_text)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)  # (G,)

        # 3. Group-normalized advantages: Â_i = (r_i - μ) / (σ + ε)
        mean_r = rewards_t.mean()
        std_r = rewards_t.std().clamp(min=1e-8)
        advantages = (rewards_t - mean_r) / std_r  # (G,)

        # 4. GRPO loss
        self.model.train()

        # Current policy log probs (with gradient)
        pi_logprobs = self._get_token_log_probs(self.model, completions, response_mask)  # (G,)

        # Old policy log probs — use rollout snapshot (captured above), NOT ref_model.
        # FIX: π_old must be the policy AT rollout time. ref_model is used for KL regularization only.
        # pi_old_logprobs already computed above during rollout (frozen).

        # KL penalty vs frozen reference model (separate from π_old)
        with torch.no_grad():
            ref_logprobs = self._get_token_log_probs(self.ref_model, completions, response_mask)

        kl_penalty = (pi_logprobs - ref_logprobs).mean()

        # Clipped PPO-style objective
        log_ratio = pi_logprobs - pi_old_logprobs.detach()
        ratio = torch.exp(log_ratio.clamp(-20, 20))
        eps = self.cfg.epsilon
        clipped_ratio = torch.clamp(ratio, 1 - eps, 1 + eps)

        # Advantage-weighted objective
        policy_loss = -torch.min(
            ratio * advantages,
            clipped_ratio * advantages,
        ).mean()

        loss = policy_loss + self.cfg.beta * kl_penalty

        return {
            "loss": loss,
            "reward_mean": mean_r.item(),
            "reward_std": std_r.item(),
            "kl": kl_penalty.item(),
        }

    def train(self, prompt_dataloader) -> None:
        """GRPO training loop."""
        self.optimizer.zero_grad()
        step = 0
        micro = 0

        for batch in prompt_dataloader:
            if step >= self.cfg.max_steps:
                break

            prompt_ids = batch["input_ids"][:1].cuda()  # process one prompt at a time
            result = self.grpo_step(prompt_ids)

            result["loss"].backward()
            micro += 1

            if micro == self.cfg.grad_accum_steps:
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                step += 1
                micro = 0

                if step % 5 == 0:
                    print(
                        f"GRPO step {step}/{self.cfg.max_steps} | "
                        f"reward_mean {result['reward_mean']:.3f} ± {result['reward_std']:.3f} | "
                        f"kl {result['kl']:.4f}"
                    )
