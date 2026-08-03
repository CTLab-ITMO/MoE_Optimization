from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Iterator, Tuple

import torch
import torch.nn as nn

from moe_opt_core.experiment import CandidateAction, CandidatePlanContract
from moe_opt_core.pipelines.h1_merge import merge_cluster_modules
from moe_opt_runner.runtime import RuntimeContext


@dataclass(frozen=True)
class AppliedTransformInfo:
    candidate_id: str
    original_experts: int
    active_experts: int
    model_parameter_mb: float


def model_parameter_mb(model: nn.Module) -> float:
    unique = {id(parameter): parameter for parameter in model.parameters()}
    total_bytes = sum(parameter.numel() * parameter.element_size() for parameter in unique.values())
    return float(total_bytes / (1024.0 * 1024.0))


def _merge_expert_group(
    experts: list[nn.Module],
    group: tuple[int, ...],
    *,
    method: str,
    rank: int,
    shrink_tau: float,
    cache: Dict[Tuple[object, ...], nn.Module] | None = None,
) -> nn.Module:
    if len(group) == 1:
        return experts[group[0]]
    cache_key = (method, group, int(rank), float(shrink_tau))
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    merged = deepcopy(experts[group[0]])
    merged.load_state_dict(
        merge_cluster_modules(
            experts=experts,
            cluster=group,
            method=method,
            rank=rank,
            shrink_tau=shrink_tau,
        )
    )
    if cache is not None:
        cache[cache_key] = merged
    return merged


def _aggregate_linear_gate(gate: nn.Linear, groups: tuple[tuple[int, ...], ...]) -> nn.Linear:
    new_gate = nn.Linear(
        gate.in_features,
        len(groups),
        bias=gate.bias is not None,
        device=gate.weight.device,
        dtype=gate.weight.dtype,
    )
    with torch.no_grad():
        for idx, group in enumerate(groups):
            source = torch.tensor(group, dtype=torch.long, device=gate.weight.device)
            new_gate.weight[idx].copy_(gate.weight.index_select(0, source).mean(dim=0))
            if gate.bias is not None:
                new_gate.bias[idx].copy_(gate.bias.index_select(0, source).mean())
    return new_gate


class GroupedLinearGate(nn.Module):
    def __init__(self, source_gate: nn.Linear, groups: tuple[tuple[int, ...], ...]) -> None:
        super().__init__()
        self.source_gate = deepcopy(source_gate)
        self.groups = groups
        self.in_features = source_gate.in_features
        self.out_features = len(groups)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        source_logits = self.source_gate(hidden)
        grouped = [
            torch.logsumexp(source_logits[..., list(group)], dim=-1)
            for group in self.groups
        ]
        return torch.stack(grouped, dim=-1)


@contextmanager
def reversible_one_layer_transform(
    ctx: RuntimeContext,
    plan: CandidatePlanContract,
    *,
    merge_method: str,
    geometry_rank: int,
    shrink_tau: float,
    expert_cache: Dict[Tuple[object, ...], nn.Module] | None = None,
) -> Iterator[AppliedTransformInfo]:
    handle = ctx.handles_by_layer.get(plan.layer)
    if handle is None:
        raise ValueError(f"Layer {plan.layer} is not available")
    moe = handle.moe
    old_experts_module = getattr(moe, "experts")
    old_gate = getattr(moe, "gate")
    if not isinstance(old_gate, nn.Linear):
        raise TypeError(f"Only linear gates are supported, got {type(old_gate)}")
    experts = list(old_experts_module)

    tracked_attributes = (
        "num_experts",
        "num_local_experts",
        "num_experts_per_tok",
        "num_experts_per_token",
        "top_k",
        "k",
    )
    old_attributes = {
        name: getattr(moe, name)
        for name in tracked_attributes
        if hasattr(moe, name)
    }
    old_handle_gate = handle.gate
    old_handle_experts = handle.experts

    method = merge_method if plan.action == CandidateAction.MERGE else "baseline"
    new_experts = nn.ModuleList(
        [
            _merge_expert_group(
                experts,
                group,
                method=method,
                rank=geometry_rank,
                shrink_tau=shrink_tau,
                cache=expert_cache,
            )
            for group in plan.groups
        ]
    )
    if plan.action == CandidateAction.MERGE:
        new_gate = GroupedLinearGate(old_gate, plan.groups)
    else:
        new_gate = _aggregate_linear_gate(old_gate, plan.groups)
    try:
        setattr(moe, "experts", new_experts)
        setattr(moe, "gate", new_gate)
        handle.experts = new_experts
        handle.gate = new_gate
        for name in ("num_experts", "num_local_experts"):
            if hasattr(moe, name):
                setattr(moe, name, len(new_experts))
        for name in ("num_experts_per_tok", "num_experts_per_token", "top_k", "k"):
            if hasattr(moe, name):
                setattr(moe, name, min(int(getattr(moe, name)), len(new_experts)))
        yield AppliedTransformInfo(
            candidate_id=plan.candidate_id,
            original_experts=len(experts),
            active_experts=len(new_experts),
            model_parameter_mb=model_parameter_mb(ctx.model),
        )
    finally:
        setattr(moe, "experts", old_experts_module)
        setattr(moe, "gate", old_gate)
        handle.experts = old_handle_experts
        handle.gate = old_handle_gate
        for name, value in old_attributes.items():
            setattr(moe, name, value)
