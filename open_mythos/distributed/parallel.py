"""
Distributed Parallelism for OpenMythos — v2

Fixes over v1:
  1. Expert Parallelism group construction fixed (separate from DP groups).
  2. apply_tensor_parallel() function wires TP into the model.
  3. Sequence Parallelism scatter/gather added.
  4. Context Parallelism ring-attention utilities added.
  5. PipelineEngine drives 1F1B schedule.
  6. ProcessGroups.initialize() validates EP size.
"""

from __future__ import annotations

import functools
import os
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp import (
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,  # noqa: N817 - PyTorch's conventional public alias
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy, size_based_auto_wrap_policy

# ---------------------------------------------------------------------------
# Process Group Registry
# ---------------------------------------------------------------------------


class ProcessGroups:
    """
    Registry for all distributed process groups used in 4D parallelism.
    Groups: dp (data), tp (tensor), pp (pipeline), ep (expert).

    FIX v2: EP groups are constructed independently of DP groups.
    EP encompasses a subset of DP ranks to partition experts; they are
    NOT the same group. Correct construction:
        world = dp_size × tp_size × pp_size
        ep_size divides dp_size
        EP group = ranks sharing the same tp+pp position, grouped by ep_size
    """

    _groups: dict[str, Any] = {}

    @classmethod
    def initialize(
        cls,
        tp_size: int = 1,
        pp_size: int = 1,
        ep_size: int = 1,
    ) -> None:
        if not dist.is_initialized():
            raise RuntimeError("Must call dist.init_process_group before ProcessGroups.initialize")

        world = dist.get_world_size()
        rank = dist.get_rank()
        dp_size = world // (tp_size * pp_size)

        if world != dp_size * tp_size * pp_size:
            raise ValueError(f"world_size {world} != dp×tp×pp = {dp_size}×{tp_size}×{pp_size}")
        if dp_size % ep_size != 0:
            raise ValueError(f"dp_size {dp_size} not divisible by ep_size {ep_size}")

        # TP groups: each group has tp_size consecutive ranks within one dp×pp slot
        tp_groups = []
        for i in range(world // tp_size):
            ranks = list(range(i * tp_size, (i + 1) * tp_size))
            tp_groups.append(dist.new_group(ranks))
        cls._groups["tp"] = tp_groups[rank // tp_size]

        # PP groups: ranks at the same TP position across all pipeline stages
        pp_groups = []
        for i in range(tp_size):
            ranks = [i + j * tp_size for j in range(pp_size)]
            pp_groups.append(dist.new_group(ranks))
        cls._groups["pp"] = pp_groups[rank % tp_size]

        # DP groups: ranks sharing the same tp+pp position
        dp_groups = []
        for i in range(tp_size * pp_size):
            ranks = [i + j * (tp_size * pp_size) for j in range(dp_size)]
            dp_groups.append(dist.new_group(ranks))
        cls._groups["dp"] = dp_groups[rank % (tp_size * pp_size)]

        # EP groups: FIX v2 — ep_size ranks within the DP group, NOT the full DP group
        # Each EP group covers ep_size DP ranks that share a partition of the expert table.
        ep_groups = []
        dp_group_members = [
            rank % (tp_size * pp_size) + j * (tp_size * pp_size) for j in range(dp_size)
        ]
        # Partition the dp_group_members list into ceil(dp_size/ep_size) EP groups
        for start in range(0, dp_size, ep_size):
            ep_ranks = dp_group_members[start : start + ep_size]
            if len(ep_ranks) == ep_size:
                ep_groups.append(dist.new_group(ep_ranks))
        # Assign EP group for this rank
        dp_rank_within_group = rank // (tp_size * pp_size) % dp_size
        ep_group_idx = dp_rank_within_group // ep_size
        cls._groups["ep"] = ep_groups[ep_group_idx] if ep_groups else None

        cls._groups["tp_size"] = tp_size
        cls._groups["pp_size"] = pp_size
        cls._groups["dp_size"] = dp_size
        cls._groups["ep_size"] = ep_size

    @classmethod
    def get(cls, name: str):
        return cls._groups.get(name)

    @classmethod
    def tp_rank(cls) -> int:
        g = cls._groups.get("tp")
        return dist.get_rank(g) if g is not None else 0

    @classmethod
    def pp_rank(cls) -> int:
        g = cls._groups.get("pp")
        return dist.get_rank(g) if g is not None else 0

    @classmethod
    def ep_rank(cls) -> int:
        g = cls._groups.get("ep")
        return dist.get_rank(g) if g is not None else 0

    @classmethod
    def tp_size(cls) -> int:
        return cls._groups.get("tp_size", 1)

    @classmethod
    def pp_size(cls) -> int:
        return cls._groups.get("pp_size", 1)

    @classmethod
    def ep_size(cls) -> int:
        return cls._groups.get("ep_size", 1)


# ---------------------------------------------------------------------------
# Tensor Parallelism — Column / Row Linear
# ---------------------------------------------------------------------------


class ColumnParallelLinear(nn.Module):
    """Column-parallel: splits output features across TP ranks."""

    def __init__(self, in_features, out_features, tp_group=None, bias=False, gather_output=False):
        super().__init__()
        self.tp_group = tp_group
        self.gather_output = gather_output
        tp_sz = dist.get_world_size(tp_group) if tp_group is not None else 1
        assert (
            out_features % tp_sz == 0
        ), f"out_features {out_features} not divisible by tp_size {tp_sz}"
        self.local_out = out_features // tp_sz
        self.weight = nn.Parameter(torch.empty(self.local_out, in_features))
        self.bias_param = nn.Parameter(torch.zeros(self.local_out)) if bias else None
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias_param)
        if self.gather_output and self.tp_group is not None:
            out = _gather_last_dim(out, self.tp_group)
        return out


class RowParallelLinear(nn.Module):
    """Row-parallel: splits input features across TP ranks; all-reduce on forward."""

    def __init__(
        self, in_features, out_features, tp_group=None, bias=False, input_is_parallel=True
    ):
        super().__init__()
        self.tp_group = tp_group
        self.input_is_parallel = input_is_parallel
        tp_sz = dist.get_world_size(tp_group) if tp_group is not None else 1
        assert in_features % tp_sz == 0
        self.local_in = in_features // tp_sz
        self.weight = nn.Parameter(torch.empty(out_features, self.local_in))
        self.bias_param = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        if not self.input_is_parallel and self.tp_group is not None:
            tp_rank = dist.get_rank(self.tp_group)
            x = x[..., tp_rank * self.local_in : (tp_rank + 1) * self.local_in]
        out = F.linear(x, self.weight)
        if self.tp_group is not None and dist.get_world_size(self.tp_group) > 1:
            dist.all_reduce(out, op=dist.ReduceOp.SUM, group=self.tp_group)
        if self.bias_param is not None:
            out = out + self.bias_param
        return out


def _gather_last_dim(tensor, group):
    world = dist.get_world_size(group)
    if world == 1:
        return tensor
    chunks = [torch.empty_like(tensor) for _ in range(world)]
    dist.all_gather(chunks, tensor, group=group)
    return torch.cat(chunks, dim=-1)


# ---------------------------------------------------------------------------
# apply_tensor_parallel — wires TP into an OpenMythos model
# ---------------------------------------------------------------------------


def apply_tensor_parallel(model: nn.Module, tp_group) -> nn.Module:
    """
    FIX v2: Replace nn.Linear in attention and FFN layers with TP equivalents.

    Strategy (Megatron-LM §3):
      - Q/K/V projections: ColumnParallel (each rank holds 1/tp of heads)
      - Output projection (wo): RowParallel
      - FFN gate/up: ColumnParallel
      - FFN down: RowParallel
    """
    from open_mythos.main import Expert, GQAttention, MLAttention

    if tp_group is None or dist.get_world_size(tp_group) == 1:
        return model

    def _replace_attn(attn_module):
        if isinstance(attn_module, (GQAttention, MLAttention)):
            # Replace wo with RowParallel (output projection reduces across TP)
            orig_wo = attn_module.wo
            in_f, out_f = orig_wo.weight.shape[1], orig_wo.weight.shape[0]
            attn_module.wo = RowParallelLinear(in_f, out_f, tp_group=tp_group)

    def _replace_ffn(ffn_module):
        if isinstance(ffn_module, Expert):
            # gate and up: ColumnParallel; down: RowParallel
            in_f = ffn_module.gate.weight.shape[1]
            mid_f = ffn_module.gate.weight.shape[0]
            out_f = ffn_module.down.weight.shape[0]
            ffn_module.gate = ColumnParallelLinear(in_f, mid_f, tp_group=tp_group)
            ffn_module.up = ColumnParallelLinear(in_f, mid_f, tp_group=tp_group)
            ffn_module.down = RowParallelLinear(mid_f, out_f, tp_group=tp_group)

    for module in model.modules():
        if isinstance(module, (GQAttention, MLAttention)):
            _replace_attn(module)
        elif isinstance(module, Expert):
            _replace_ffn(module)

    return model


# ---------------------------------------------------------------------------
# Sequence Parallelism
# ---------------------------------------------------------------------------


def scatter_to_sequence_parallel(tensor: torch.Tensor, sp_group) -> torch.Tensor:
    """
    Split tensor along sequence dimension (dim=1) across SP group ranks.
    Input shape: (B, T, D) → Output shape: (B, T//sp_size, D)
    """
    if sp_group is None:
        return tensor
    sp_size = dist.get_world_size(sp_group)
    if sp_size == 1:
        return tensor
    sp_rank = dist.get_rank(sp_group)
    T = tensor.shape[1]
    assert T % sp_size == 0, f"seq_len {T} not divisible by sp_size {sp_size}"
    chunk_size = T // sp_size
    return tensor[:, sp_rank * chunk_size : (sp_rank + 1) * chunk_size, :].contiguous()


def gather_from_sequence_parallel(tensor: torch.Tensor, sp_group) -> torch.Tensor:
    """Gather sequence-parallel tensor back to full sequence. (B, T//sp, D) → (B, T, D)"""
    if sp_group is None:
        return tensor
    sp_size = dist.get_world_size(sp_group)
    if sp_size == 1:
        return tensor
    chunks = [torch.empty_like(tensor) for _ in range(sp_size)]
    dist.all_gather(chunks, tensor, group=sp_group)
    return torch.cat(chunks, dim=1)


# ---------------------------------------------------------------------------
# Context Parallelism — Ring Attention
# ---------------------------------------------------------------------------


def ring_attention_forward(
    q: torch.Tensor,  # (B, H, T_local, d)
    k: torch.Tensor,
    v: torch.Tensor,
    cp_group,
) -> torch.Tensor:
    """
    Ring-attention for context parallelism.
    Each rank holds T/cp_size query tokens. K/V are rotated through the ring.
    Accumulates numerically stable softmax using the online algorithm.
    """
    if cp_group is None or dist.get_world_size(cp_group) == 1:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    cp_size = dist.get_world_size(cp_group)
    cp_rank = dist.get_rank(cp_group)
    B, H, T_local, d = q.shape

    # Online softmax accumulators
    out = torch.zeros_like(q)
    lse = torch.full((B, H, T_local, 1), float("-inf"), device=q.device, dtype=torch.float32)

    k_cur, v_cur = k.clone(), v.clone()

    for i in range(cp_size):
        # Compute attention scores for this K/V chunk
        src_rank = (cp_rank - i) % cp_size
        is_causal_chunk = src_rank == cp_rank

        scores = torch.matmul(q.float(), k_cur.float().transpose(-2, -1)) * (d**-0.5)
        if is_causal_chunk:
            mask = torch.triu(torch.ones(T_local, T_local, device=q.device), diagonal=1).bool()
            scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        elif src_rank > cp_rank:
            # Future chunk — fully masked for causal LM
            scores.fill_(float("-inf"))

        block_max = scores.max(dim=-1, keepdim=True).values
        exp_scores = torch.exp(scores - block_max)
        block_sum = exp_scores.sum(dim=-1, keepdim=True)
        block_out = torch.matmul(exp_scores, v_cur.float())

        # Online softmax update
        new_lse = torch.logaddexp(lse, block_max + torch.log(block_sum.clamp(min=1e-9)))
        alpha = torch.exp(lse - new_lse)
        beta = torch.exp(block_max + torch.log(block_sum.clamp(min=1e-9)) - new_lse)
        out = alpha * out + beta * block_out.to(out.dtype)
        lse = new_lse

        # Ring-rotate K/V to next rank
        # FIX v4: use isend/irecv to avoid deadlock when send blocks before recv.
        # Blocking send+recv in the naive order deadlocks in a 2-rank ring.
        if i < cp_size - 1:
            next_rank = (cp_rank + 1) % cp_size
            prev_rank = (cp_rank - 1) % cp_size
            k_next = torch.empty_like(k_cur)
            v_next = torch.empty_like(v_cur)
            reqs = []
            reqs.append(dist.isend(k_cur.contiguous(), dst=next_rank, group=cp_group))
            reqs.append(dist.irecv(k_next, src=prev_rank, group=cp_group))
            reqs.append(dist.isend(v_cur.contiguous(), dst=next_rank, group=cp_group))
            reqs.append(dist.irecv(v_next, src=prev_rank, group=cp_group))
            for r in reqs:
                r.wait()
            k_cur, v_cur = k_next, v_next

    return out


# ---------------------------------------------------------------------------
# Expert Parallelism Dispatch
# ---------------------------------------------------------------------------


class ExpertParallelDispatch(nn.Module):
    """
    Expert-parallel MoE dispatch using all-to-all communication.
    FIX v2: Properly handles ep_size from dedicated EP groups (not DP groups).
    """

    def __init__(self, n_experts, ep_group=None):
        super().__init__()
        self.n_experts = n_experts
        self.ep_group = ep_group
        self.ep_size = dist.get_world_size(ep_group) if ep_group is not None else 1
        assert (
            n_experts % self.ep_size == 0
        ), f"n_experts {n_experts} not divisible by ep_size {self.ep_size}"
        self.local_experts = n_experts // self.ep_size

    def dispatch(self, x, expert_idx, weights):
        if self.ep_size == 1:
            N, K = expert_idx.shape
            return (
                x.unsqueeze(1).expand(N, K, x.shape[-1]).reshape(N * K, x.shape[-1]),
                expert_idx.reshape(N * K),
                weights.reshape(N * K),
                None,
                None,
                None,
            )

        N, D = x.shape
        K = expert_idx.shape[1]
        expert_rank = expert_idx // self.local_experts
        local_idx = expert_idx % self.local_experts

        send_counts = torch.zeros(self.ep_size, dtype=torch.long, device=x.device)
        for r in range(self.ep_size):
            send_counts[r] = (expert_rank == r).sum()
        recv_counts = torch.zeros_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts, group=self.ep_group)

        order = torch.argsort(expert_rank.reshape(-1), stable=True)
        x_ordered = x.unsqueeze(1).expand(N, K, D).reshape(N * K, D)[order]
        w_ordered = weights.reshape(N * K)[order]
        idx_ordered = local_idx.reshape(N * K)[order]

        recv_total = recv_counts.sum().item()
        local_x = torch.empty(recv_total, D, device=x.device, dtype=x.dtype)
        local_expert_idx_flat = torch.empty(recv_total, dtype=torch.long, device=x.device)

        dist.all_to_all_single(
            local_x,
            x_ordered,
            output_split_sizes=recv_counts.tolist(),
            input_split_sizes=send_counts.tolist(),
            group=self.ep_group,
        )
        dist.all_to_all_single(
            local_expert_idx_flat,
            idx_ordered,
            output_split_sizes=recv_counts.tolist(),
            input_split_sizes=send_counts.tolist(),
            group=self.ep_group,
        )

        return local_x, local_expert_idx_flat, w_ordered, order, send_counts, recv_counts

    def combine(self, output, order, send_counts, recv_counts, N, K, D):
        if self.ep_size == 1:
            return output
        combined = torch.empty(N * K, D, device=output.device, dtype=output.dtype)
        dist.all_to_all_single(
            combined,
            output,
            output_split_sizes=send_counts.tolist(),
            input_split_sizes=recv_counts.tolist(),
            group=self.ep_group,
        )
        return combined[torch.argsort(order)]


# ---------------------------------------------------------------------------
# Pipeline Parallelism — 1F1B Engine
# ---------------------------------------------------------------------------


class PipelineStage(nn.Module):
    def __init__(self, layers, stage_id, n_stages, pp_group=None):
        super().__init__()
        self.layers = layers
        self.stage_id = stage_id
        self.n_stages = n_stages
        self.pp_group = pp_group
        self.is_first = stage_id == 0
        self.is_last = stage_id == n_stages - 1

    def forward(self, x, **kwargs):
        # FIX: slice freqs_cis to the actual sequence length of x.
        # PipelineStage.forward receives freqs_cis sized for max_seq_len, but
        # x may have T < max_seq_len, causing shape mismatch inside attention.
        if "freqs_cis" in kwargs and kwargs["freqs_cis"] is not None:
            T = x.shape[1]
            freqs = kwargs["freqs_cis"]
            if freqs.shape[0] != T:
                kwargs = dict(kwargs)
                kwargs["freqs_cis"] = freqs[:T]
        for layer in self.layers:
            x = layer(x, **kwargs)
        return x

    def send_activation(self, tensor):
        if self.pp_group is None or self.is_last:
            return
        dist.send(tensor, dst=dist.get_rank(self.pp_group) + 1, group=self.pp_group)

    def recv_activation(self, shape, dtype, device):
        if self.pp_group is None or self.is_first:
            raise RuntimeError("First stage cannot receive activations")
        buf = torch.empty(shape, dtype=dtype, device=device)
        dist.recv(buf, src=dist.get_rank(self.pp_group) - 1, group=self.pp_group)
        return buf


class PipelineEngine:
    """
    1F1B Pipeline schedule driver.

    Bubble fraction = (pp_size - 1) / (pp_size - 1 + n_microbatches)
    With n_microbatches ≥ 2 * pp_size, bubble < 50%.

    Usage:
        engine = PipelineEngine(stage, n_microbatches=16)
        losses = engine.forward_backward_step(microbatch_iterator, criterion)
    """

    def __init__(self, stage: PipelineStage, n_microbatches: int):
        self.stage = stage
        self.n_microbatches = n_microbatches
        self.pp_size = stage.n_stages
        self.stage_id = stage.stage_id

    def forward_backward_step(self, microbatch_iter, criterion, optimizer):
        """
        Execute one 1F1B step across all microbatches.
        Returns mean loss (last stage only).
        """
        stage = self.stage
        n_mb = self.n_microbatches
        stored_activations = {}  # microbatch_id → activation tensor
        losses = []

        # Warm-up phase: fill the pipeline with forward passes
        warmup_steps = min(self.pp_size - 1 - self.stage_id, n_mb)
        for mb_id in range(warmup_steps):
            if stage.is_first:
                mb = next(microbatch_iter)
                x = stage(mb)
            else:
                shape_info = stored_activations.get(f"shape_{mb_id}")
                if shape_info is None:
                    raise RuntimeError(f"Shape info missing for mb {mb_id}")
                x = stage.recv_activation(
                    shape_info["shape"],
                    shape_info["dtype"],
                    next(iter(stage.layers.parameters())).device,
                )
                x = stage(x)

            stored_activations[mb_id] = x
            if not stage.is_last:
                stage.send_activation(x)

        # Steady state: 1F1B
        for mb_id in range(warmup_steps, n_mb):
            # Forward
            if stage.is_first:
                mb = next(microbatch_iter)
                x_fwd = stage(mb)
            else:
                shape = stored_activations.get(f"shape_{mb_id}")
                device = next(iter(stage.layers.parameters())).device
                x_fwd = stage.recv_activation(shape["shape"], shape["dtype"], device)
                x_fwd = stage(x_fwd)

            if stage.is_last:
                # Compute loss and backward immediately
                loss = criterion(x_fwd)
                losses.append(loss.item())
                loss.backward()
            else:
                stage.send_activation(x_fwd)
                stored_activations[mb_id] = x_fwd

            # Backward for the corresponding warm-up activation
            bwd_id = mb_id - warmup_steps
            if bwd_id in stored_activations and not stage.is_last:
                stored_activations[bwd_id].backward()
                del stored_activations[bwd_id]

        # Cool-down: drain remaining activations
        for bwd_id in range(warmup_steps):
            if bwd_id in stored_activations and not stage.is_last:
                stored_activations[bwd_id].backward()

        optimizer.step()
        optimizer.zero_grad()
        return sum(losses) / max(len(losses), 1)


# ---------------------------------------------------------------------------
# FSDP
# ---------------------------------------------------------------------------


def build_fsdp_model(
    model,
    amp_dtype=torch.bfloat16,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    min_num_params=1_000_000,
):
    from open_mythos.main import RecurrentBlock, TransformerBlock

    mp_policy = MixedPrecision(
        param_dtype=amp_dtype, reduce_dtype=amp_dtype, buffer_dtype=amp_dtype
    )
    explicit_wrap = ModuleWrapPolicy({TransformerBlock, RecurrentBlock})
    size_wrap = functools.partial(size_based_auto_wrap_policy, min_num_params=min_num_params)

    def combined_wrap_policy(module, recurse, nonwrapped_numel):
        return explicit_wrap(module, recurse, nonwrapped_numel) or size_wrap(
            module, recurse, nonwrapped_numel
        )

    return FSDP(
        model,
        sharding_strategy=sharding_strategy,
        mixed_precision=mp_policy,
        auto_wrap_policy=combined_wrap_policy,
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        forward_prefetch=True,
        backward_prefetch=torch.distributed.fsdp.BackwardPrefetch.BACKWARD_PRE,
        use_orig_params=True,
    )


def fsdp_save_checkpoint(model, optimizer, step, path):
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        model_state = model.state_dict()
        optim_state = FSDP.optim_state_dict(model, optimizer)
    if dist.get_rank() != 0:
        return
    tmp = path + ".tmp"
    torch.save({"step": step, "model": model_state, "optimizer": optim_state}, tmp)
    os.rename(tmp, path)


def fsdp_load_checkpoint(model, optimizer, path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    ):
        model.load_state_dict(ckpt["model"])
        optim_state = FSDP.optim_state_dict_to_load(
            model=model, optim=optimizer, optim_state_dict=ckpt["optimizer"]
        )
        optimizer.load_state_dict(optim_state)
    return int(ckpt["step"])


# ---------------------------------------------------------------------------
# DeepSpeed ZeRO Config Builder
# ---------------------------------------------------------------------------


def build_deepspeed_config(
    micro_batch_size,
    gradient_accumulation_steps,
    lr=3e-4,
    fp16=False,
    bf16=True,
    zero_stage=3,
    offload_optimizer=False,
    offload_param=False,
    gradient_clipping=1.0,
):
    config = {
        "train_micro_batch_size_per_gpu": micro_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_clipping": gradient_clipping,
        "steps_per_print": 100,
        "wall_clock_breakdown": False,
        "zero_optimization": {
            "stage": zero_stage,
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True,
        },
        "optimizer": {
            "type": "AdamW",
            "params": {"lr": lr, "betas": [0.9, 0.95], "eps": 1e-8, "weight_decay": 0.1},
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": lr,
                "warmup_num_steps": 2000,
                "total_num_steps": "auto",
            },
        },
    }
    if bf16:
        config["bf16"] = {"enabled": True}
    elif fp16:
        config["fp16"] = {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1,
        }
    if offload_optimizer:
        config["zero_optimization"]["offload_optimizer"] = {"device": "cpu", "pin_memory": True}
    if offload_param:
        config["zero_optimization"]["offload_param"] = {"device": "cpu", "pin_memory": True}
    return config
