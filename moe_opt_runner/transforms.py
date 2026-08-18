from __future__ import annotations

from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Iterator, Sequence, Tuple

import torch
import torch.nn as nn

from moe_opt_core.experiment import CandidateAction, CandidatePlanContract
from moe_opt_core.experiment.candidates import validate_candidate_plan
from moe_opt_core.pipelines.h1_merge import merge_cluster_modules
from moe_opt_runner.runtime import RuntimeContext


@dataclass(frozen=True)
class AppliedTransformInfo:
    candidate_id: str
    original_experts: int
    active_experts: int
    model_parameter_mb: float
    shared_experts_preserved: int
    merge_weighting: str


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
    merge_weights: tuple[float, ...] | None = None,
    cache: Dict[Tuple[object, ...], nn.Module] | None = None,
) -> nn.Module:
    if len(group) == 1:
        return experts[group[0]]
    cache_key = (method, group, int(rank), float(shrink_tau), merge_weights)
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
            usage=merge_weights,
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
    validate_candidate_plan(plan, num_experts=len(experts))
    runtime_contract = ctx.runtime_contracts_by_layer.get(plan.layer)
    if runtime_contract is not None and runtime_contract.routed_experts != len(experts):
        raise ValueError("Runtime contract does not match the routed expert collection")
    shared_expert = getattr(moe, "shared_expert", None)
    shared_expert_gate = getattr(moe, "shared_expert_gate", None)

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
                merge_weights=(
                    plan.group_merge_weights[index]
                    if plan.group_merge_weights
                    else None
                ),
                cache=expert_cache,
            )
            for index, group in enumerate(plan.groups)
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
            shared_experts_preserved=(runtime_contract.shared_experts if runtime_contract is not None else 0),
            merge_weighting=plan.merge_weighting.value,
        )
    finally:
        setattr(moe, "experts", old_experts_module)
        setattr(moe, "gate", old_gate)
        handle.experts = old_handle_experts
        handle.gate = old_handle_gate
        for name, value in old_attributes.items():
            setattr(moe, name, value)
        if getattr(moe, "shared_expert", None) is not shared_expert:
            raise RuntimeError("Shared expert changed during a routed-expert transform")
        if getattr(moe, "shared_expert_gate", None) is not shared_expert_gate:
            raise RuntimeError("Shared expert gate changed during a routed-expert transform")


@contextmanager
def reversible_multi_layer_transform(
    ctx: RuntimeContext,
    plans: Sequence[CandidatePlanContract],
    *,
    merge_method: str,
    geometry_rank: int,
    shrink_tau: float,
) -> Iterator[tuple[AppliedTransformInfo, ...]]:
    layers = tuple(plan.layer for plan in plans)
    if len(set(layers)) != len(layers):
        raise ValueError("A multi-layer transform may contain only one plan per layer")
    with ExitStack() as stack:
        transforms = tuple(
            stack.enter_context(
                reversible_one_layer_transform(
                    ctx,
                    plan,
                    merge_method=merge_method,
                    geometry_rank=geometry_rank,
                    shrink_tau=shrink_tau,
                    expert_cache={},
                )
            )
            for plan in plans
        )
        yield transforms


def apply_one_layer_transform_inplace(
    ctx: RuntimeContext,
    plan: CandidatePlanContract,
    *,
    merge_method: str,
    geometry_rank: int,
    shrink_tau: float,
) -> AppliedTransformInfo:
    handle = ctx.handles_by_layer.get(plan.layer)
    if handle is None:
        raise ValueError(f"Layer {plan.layer} is not available")
    moe = handle.moe
    old_gate = getattr(moe, "gate")
    if not isinstance(old_gate, nn.Linear):
        raise TypeError(f"Only linear gates are supported, got {type(old_gate)}")
    experts = list(getattr(moe, "experts"))
    validate_candidate_plan(plan, num_experts=len(experts))
    runtime_contract = ctx.runtime_contracts_by_layer.get(plan.layer)
    method = merge_method if plan.action == CandidateAction.MERGE else "baseline"
    new_experts = nn.ModuleList(
        [
            _merge_expert_group(
                experts,
                group,
                method=method,
                rank=geometry_rank,
                shrink_tau=shrink_tau,
                merge_weights=(
                    plan.group_merge_weights[index]
                    if plan.group_merge_weights
                    else None
                ),
            )
            for index, group in enumerate(plan.groups)
        ]
    )
    new_gate = (
        GroupedLinearGate(old_gate, plan.groups)
        if plan.action == CandidateAction.MERGE
        else _aggregate_linear_gate(old_gate, plan.groups)
    )
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
    return AppliedTransformInfo(
        candidate_id=plan.candidate_id,
        original_experts=len(experts),
        active_experts=len(new_experts),
        model_parameter_mb=model_parameter_mb(ctx.model),
        shared_experts_preserved=(
            runtime_contract.shared_experts if runtime_contract is not None else 0
        ),
        merge_weighting=plan.merge_weighting.value,
    )


def apply_multi_layer_transform_inplace(
    ctx: RuntimeContext,
    plans: Sequence[CandidatePlanContract],
    *,
    merge_method: str,
    geometry_rank: int,
    shrink_tau: float,
) -> tuple[AppliedTransformInfo, ...]:
    layers = tuple(plan.layer for plan in plans)
    if len(set(layers)) != len(layers):
        raise ValueError("A multi-layer transform may contain only one plan per layer")
    return tuple(
        apply_one_layer_transform_inplace(
            ctx,
            plan,
            merge_method=merge_method,
            geometry_rank=geometry_rank,
            shrink_tau=shrink_tau,
        )
        for plan in plans
    )
