from __future__ import annotations

import pytest

from moe_opt_core.experiment import (
    ReplicatedRecipeObservationContract,
    SensitivityVariantObservationContract,
)
from moe_opt_core.experiment.campaign_analysis import build_campaign_summary


def _recipe(seed: int, degradation: float) -> ReplicatedRecipeObservationContract:
    return ReplicatedRecipeObservationContract(
        seed=seed,
        recipe_id="rmt-n3",
        record_type="selected_multilayer_candidate",
        methods=("rmt_functional_usage",),
        layer_count=3,
        target_compression=0.25,
        achieved_compression=0.25,
        relative_perplexity_degradation=degradation,
        throughput_gain=0.01 * seed,
        memory_reduction=0.1,
    )


def test_campaign_summary_aggregates_seeds_and_reports_missing_runs() -> None:
    summary = build_campaign_summary(
        config_name="campaign",
        expected_seeds=(1, 2, 3),
        completed_seeds=(1, 2),
        recipe_observations=(_recipe(1, 0.01), _recipe(2, 0.03)),
        sensitivity_observations=(),
        quality_budget=0.02,
    )
    assert summary.missing_seeds == (3,)
    recipe = summary.recipe_summaries[0]
    assert recipe.degradation.median == pytest.approx(0.02)
    assert recipe.degradation.p05 == pytest.approx(0.011)
    assert recipe.degradation.p95 == pytest.approx(0.029)
    assert recipe.quality_pass_rate == pytest.approx(0.5)


def test_sensitivity_summary_counts_winning_rank_and_merge_policy() -> None:
    observations = (
        SensitivityVariantObservationContract(
            1, 11, "rmt_functional_usage", 0.25, "r1-uniform", 1, "uniform", 0.03
        ),
        SensitivityVariantObservationContract(
            1, 11, "rmt_functional_usage", 0.25, "r4-ridge", 4, "functional_ridge", 0.01
        ),
        SensitivityVariantObservationContract(
            2, 11, "rmt_functional_usage", 0.25, "r4-uniform", 4, "uniform", 0.02
        ),
    )
    summary = build_campaign_summary(
        config_name="campaign",
        expected_seeds=(1, 2),
        completed_seeds=(1, 2),
        recipe_observations=(),
        sensitivity_observations=observations,
        quality_budget=0.05,
    )
    cell = summary.sensitivity_summaries[0]
    assert cell.spectral_rank_counts == ((4, 2),)
    assert cell.merge_weighting_counts == (("functional_ridge", 1), ("uniform", 1))
    assert cell.robust_score.median == pytest.approx(0.015)
