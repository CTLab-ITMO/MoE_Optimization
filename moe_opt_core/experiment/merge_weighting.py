from __future__ import annotations

import math
from typing import Sequence

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

    result: list[tuple[float, ...]] = []
    for raw_group in groups:
        group = tuple(int(index) for index in raw_group)
        if not group:
            raise ValueError("expert groups must not be empty")
        if any(index < 0 or index >= len(usage) for index in group):
            raise ValueError("expert group contains an invalid usage index")

        if policy == MergeWeightingPolicy.UNIFORM:
            weights = tuple(1.0 / len(group) for _ in group)
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
