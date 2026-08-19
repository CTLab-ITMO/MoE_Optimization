from __future__ import annotations

from collections import Counter, defaultdict
import math
import statistics
from typing import Sequence

from moe_opt_core.experiment.contracts import (
    MultiLayerCampaignSummaryContract,
    ReplicatedRecipeObservationContract,
    ReplicatedRecipeSummaryContract,
    ScalarDistributionContract,
    SensitivityCellSummaryContract,
    SensitivityVariantObservationContract,
)


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = tuple(sorted(float(value) for value in values))
    if not ordered:
        raise ValueError("values must not be empty")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_scalar_distribution(
    values: Sequence[float],
) -> ScalarDistributionContract:
    normalized = tuple(float(value) for value in values)
    if not normalized:
        raise ValueError("values must not be empty")
    return ScalarDistributionContract(
        count=len(normalized),
        median=float(statistics.median(normalized)),
        p05=_quantile(normalized, 0.05),
        p95=_quantile(normalized, 0.95),
        minimum=min(normalized),
        maximum=max(normalized),
    )


def summarize_replicated_recipes(
    observations: Sequence[ReplicatedRecipeObservationContract],
    *,
    quality_budget: float,
) -> tuple[ReplicatedRecipeSummaryContract, ...]:
    if quality_budget < 0.0:
        raise ValueError("quality_budget must be non-negative")
    keys = tuple(
        (observation.seed, observation.record_type, observation.recipe_id)
        for observation in observations
    )
    if len(set(keys)) != len(keys):
        raise ValueError("replicated recipe observations must be unique per seed")
    grouped: dict[
        tuple[str, str], list[ReplicatedRecipeObservationContract]
    ] = defaultdict(list)
    for observation in observations:
        grouped[(observation.record_type, observation.recipe_id)].append(observation)

    summaries: list[ReplicatedRecipeSummaryContract] = []
    for (record_type, recipe_id), values in sorted(grouped.items()):
        ordered = tuple(sorted(values, key=lambda item: item.seed))
        first = ordered[0]
        metadata = {
            (
                item.methods,
                item.layer_count,
                round(item.target_compression, 12),
                round(item.achieved_compression, 12),
            )
            for item in ordered
        }
        if len(metadata) != 1:
            raise ValueError(f"inconsistent recipe metadata for {recipe_id}")
        degradation = tuple(
            item.relative_perplexity_degradation for item in ordered
        )
        summaries.append(
            ReplicatedRecipeSummaryContract(
                recipe_id=recipe_id,
                record_type=record_type,
                methods=first.methods,
                layer_count=first.layer_count,
                target_compression=first.target_compression,
                achieved_compression=first.achieved_compression,
                seeds=tuple(item.seed for item in ordered),
                degradation=summarize_scalar_distribution(degradation),
                throughput_gain=summarize_scalar_distribution(
                    tuple(item.throughput_gain for item in ordered)
                ),
                memory_reduction=summarize_scalar_distribution(
                    tuple(item.memory_reduction for item in ordered)
                ),
                quality_pass_rate=(
                    sum(value <= quality_budget for value in degradation)
                    / len(degradation)
                ),
            )
        )
    return tuple(summaries)


def summarize_sensitivity_winners(
    observations: Sequence[SensitivityVariantObservationContract],
) -> tuple[SensitivityCellSummaryContract, ...]:
    per_seed: dict[
        tuple[int, int, str, float], list[SensitivityVariantObservationContract]
    ] = defaultdict(list)
    for observation in observations:
        per_seed[
            (
                observation.seed,
                observation.layer,
                observation.method,
                round(observation.compression_fraction, 12),
            )
        ].append(observation)
    winners = tuple(
        min(values, key=lambda item: (item.robust_score, item.candidate_id))
        for _, values in sorted(per_seed.items())
    )
    grouped: dict[
        tuple[int, str, float], list[SensitivityVariantObservationContract]
    ] = defaultdict(list)
    for winner in winners:
        grouped[
            (
                winner.layer,
                winner.method,
                round(winner.compression_fraction, 12),
            )
        ].append(winner)

    summaries: list[SensitivityCellSummaryContract] = []
    for (layer, method, fraction), values in sorted(grouped.items()):
        ordered = tuple(sorted(values, key=lambda item: item.seed))
        rank_counts = Counter(
            item.spectral_rank
            for item in ordered
            if item.spectral_rank is not None
        )
        weighting_counts = Counter(item.merge_weighting for item in ordered)
        summaries.append(
            SensitivityCellSummaryContract(
                layer=layer,
                method=method,
                compression_fraction=fraction,
                seeds=tuple(item.seed for item in ordered),
                robust_score=summarize_scalar_distribution(
                    tuple(item.robust_score for item in ordered)
                ),
                winning_candidate_ids=tuple(item.candidate_id for item in ordered),
                spectral_rank_counts=tuple(sorted(rank_counts.items())),
                merge_weighting_counts=tuple(sorted(weighting_counts.items())),
            )
        )
    return tuple(summaries)


def build_campaign_summary(
    *,
    config_name: str,
    expected_seeds: Sequence[int],
    completed_seeds: Sequence[int],
    recipe_observations: Sequence[ReplicatedRecipeObservationContract],
    sensitivity_observations: Sequence[SensitivityVariantObservationContract],
    quality_budget: float,
) -> MultiLayerCampaignSummaryContract:
    expected = tuple(dict.fromkeys(int(seed) for seed in expected_seeds))
    completed = tuple(sorted(set(int(seed) for seed in completed_seeds)))
    unknown = sorted(set(completed) - set(expected))
    if unknown:
        raise ValueError(f"completed seeds are not configured: {unknown}")
    return MultiLayerCampaignSummaryContract(
        schema_version="1.0",
        config_name=str(config_name),
        expected_seeds=expected,
        completed_seeds=completed,
        missing_seeds=tuple(seed for seed in expected if seed not in completed),
        recipe_summaries=summarize_replicated_recipes(
            recipe_observations,
            quality_budget=quality_budget,
        ),
        sensitivity_summaries=summarize_sensitivity_winners(
            sensitivity_observations
        ),
    )
