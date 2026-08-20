from __future__ import annotations

import math
from typing import Sequence

import torch

from moe_opt_core.experiment.contracts import MergeWeightingPolicy


def _normalize_non_negative(values: Sequence[float]) -> tuple[float, ...]:
    if not values:
        raise ValueError("merge weight inputs must not be empty")
    normalized_values = tuple(float(value) for value in values)
    if any(not math.isfinite(value) or value < 0.0 for value in normalized_values):
        raise ValueError("merge weight inputs must be finite and non-negative")
    total = sum(normalized_values)
    if total <= 0.0:
        return tuple(1.0 / len(normalized_values) for _ in normalized_values)
    return tuple(value / total for value in normalized_values)


def build_group_merge_weights(
    groups: Sequence[Sequence[int]],
    expert_usage: Sequence[float],
    *,
    policy: MergeWeightingPolicy,
    minimum_weight: float = 0.0,
    ridge_lambda: float = 0.0,
    functional_embedding: torch.Tensor | None = None,
    routing_probs: torch.Tensor | None = None,
) -> tuple[tuple[float, ...], ...]:
    usage = tuple(float(value) for value in expert_usage)
    if any(not math.isfinite(value) or value < 0.0 for value in usage):
        raise ValueError("expert usage values must be finite and non-negative")
    if not 0.0 <= float(minimum_weight) < 0.5:
        raise ValueError("minimum_weight must be in [0, 0.5)")
    if policy == MergeWeightingPolicy.USAGE_BOUNDED and minimum_weight <= 0.0:
        raise ValueError("usage_bounded requires a positive minimum_weight")
    if policy != MergeWeightingPolicy.USAGE_BOUNDED and minimum_weight != 0.0:
        raise ValueError("minimum_weight is only valid for usage_bounded")
    if ridge_lambda < 0.0:
        raise ValueError("ridge_lambda must be non-negative")
    if policy == MergeWeightingPolicy.FUNCTIONAL_RIDGE:
        if ridge_lambda <= 0.0:
            raise ValueError("functional_ridge requires a positive ridge_lambda")
        if functional_embedding is None or routing_probs is None:
            raise ValueError(
                "functional_ridge requires functional_embedding and routing_probs"
            )
    elif ridge_lambda != 0.0:
        raise ValueError("ridge_lambda is only valid for functional_ridge")

    result: list[tuple[float, ...]] = []
    for raw_group in groups:
        group = tuple(int(index) for index in raw_group)
        if not group:
            raise ValueError("expert groups must not be empty")
        if any(index < 0 or index >= len(usage) for index in group):
            raise ValueError("expert group contains an invalid usage index")

        if policy == MergeWeightingPolicy.UNIFORM:
            weights = tuple(1.0 / len(group) for _ in group)
        elif policy == MergeWeightingPolicy.FUNCTIONAL_RIDGE:
            weights = _functional_ridge_group_weights(
                group,
                usage,
                functional_embedding=functional_embedding,
                routing_probs=routing_probs,
                ridge_lambda=ridge_lambda,
            )
        else:
            weights = _normalize_non_negative(tuple(usage[index] for index in group))
            if policy == MergeWeightingPolicy.USAGE_BOUNDED:
                if len(group) > 1:
                    floor = float(minimum_weight)
                    if floor * len(group) >= 1.0:
                        raise ValueError("minimum_weight is too large for the expert group")
                    residual = 1.0 - floor * len(group)
                    weights = tuple(floor + residual * weight for weight in weights)
            elif policy != MergeWeightingPolicy.USAGE_PROPORTIONAL:
                raise ValueError(f"Unsupported merge weighting policy: {policy.value}")
        result.append(weights)
    return tuple(result)


def functional_ridge_token_aligned(
    functional_embedding: torch.Tensor,
    routing_probs: torch.Tensor,
) -> bool:
    if functional_embedding.ndim != 2 or routing_probs.ndim != 2:
        raise ValueError("functional ridge inputs must be two-dimensional")
    if functional_embedding.shape[0] != routing_probs.shape[1]:
        raise ValueError("functional embedding and routing probabilities must align")
    token_count = int(routing_probs.shape[0])
    return token_count > 0 and functional_embedding.shape[1] % token_count == 0


def _functional_ridge_group_weights(
    group: tuple[int, ...],
    expert_usage: tuple[float, ...],
    *,
    functional_embedding: torch.Tensor,
    routing_probs: torch.Tensor,
    ridge_lambda: float,
) -> tuple[float, ...]:
    if len(group) == 1:
        return (1.0,)
    token_aligned = functional_ridge_token_aligned(
        functional_embedding,
        routing_probs,
    )

    indices = torch.tensor(group, dtype=torch.long)
    experts = functional_embedding.detach().float().cpu().index_select(0, indices)
    routes = routing_probs.detach().float().cpu().index_select(1, indices)
    prior = torch.tensor(
        _normalize_non_negative(tuple(expert_usage[index] for index in group)),
        dtype=torch.float64,
    )
    route_total = routes.sum(dim=1, keepdim=True)
    conditional = routes / route_total.clamp_min(1e-12)
    empty = route_total.squeeze(1) <= 1e-12
    if empty.any():
        conditional[empty] = prior.to(dtype=conditional.dtype)

    design = experts.t().to(torch.float64)
    if token_aligned:
        token_count = int(routes.shape[0])
        projection_dim = int(experts.shape[1] // token_count)
        expert_outputs = experts.reshape(len(group), token_count, projection_dim)
        target = (
            conditional.unsqueeze(-1)
            * expert_outputs.permute(1, 0, 2)
        ).sum(dim=1).reshape(-1).to(torch.float64)
        penalty_center = prior
    else:
        # Weight sketches have no token axis. Reconstruct their routing-weighted
        # centroid and use ordinary ridge shrinkage in the sketch feature space.
        route_mass = routes.sum(dim=0).to(torch.float64)
        if float(route_mass.sum().item()) <= 1e-12:
            route_prior = prior
        else:
            route_prior = route_mass / route_mass.sum()
        target = design @ route_prior
        penalty_center = torch.zeros_like(prior)

    gram = design.t() @ design
    scale = max(float(torch.trace(gram).item()) / len(group), 1e-12)
    regularization = float(ridge_lambda) * scale
    system = gram + regularization * torch.eye(len(group), dtype=torch.float64)
    rhs = design.t() @ target + regularization * penalty_center
    try:
        fitted = torch.linalg.solve(system, rhs)
    except RuntimeError:
        fitted = torch.linalg.lstsq(system, rhs.unsqueeze(1)).solution.squeeze(1)
    fitted = fitted.clamp_min(0.0)
    if float(fitted.sum().item()) <= 1e-12:
        fitted = prior
    fitted = fitted / fitted.sum()
    return tuple(float(value) for value in fitted.tolist())


def merge_weight_diagnostics(
    group_weights: Sequence[Sequence[float]],
) -> tuple[tuple[str, float], ...]:
    merged_groups = tuple(tuple(float(value) for value in group) for group in group_weights if len(group) > 1)
    if not merged_groups:
        return (
            ("mean_min_merge_weight", 1.0),
            ("mean_max_merge_weight", 1.0),
            ("mean_normalized_merge_weight_entropy", 1.0),
        )

    minimums = tuple(min(group) for group in merged_groups)
    maximums = tuple(max(group) for group in merged_groups)
    entropies = tuple(
        -sum(weight * math.log(max(weight, 1e-12)) for weight in group) / math.log(len(group))
        for group in merged_groups
    )
    return (
        ("mean_min_merge_weight", sum(minimums) / len(minimums)),
        ("mean_max_merge_weight", sum(maximums) / len(maximums)),
        ("mean_normalized_merge_weight_entropy", sum(entropies) / len(entropies)),
    )
