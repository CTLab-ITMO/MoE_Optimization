from __future__ import annotations

from moe_opt_core.experiment import (
    CandidateAction,
    CandidateMethod,
    CandidatePlanContract,
    LayerCandidateScoreContract,
)
from moe_opt_core.experiment.multilayer_planning import (
    allocate_global_budget,
    build_cumulative_recipes,
    cumulative_layer_prefixes,
    representative_layer_order,
    resolve_sensitivity_layers,
    score_layer_plans,
    select_layer_winners,
    next_multilayer_step,
)
from moe_opt_core.experiment import MultiLayerStep


def _plan(layer: int, method: str, fraction: float, suffix: str = ""):
    return CandidatePlanContract(
        candidate_id=f"l{layer}-{method}-f{fraction:.2f}{suffix}",
        layer=layer,
        method=CandidateMethod(method),
        action=(
            CandidateAction.PRUNE
            if method == "usage_prune"
            else CandidateAction.MERGE
        ),
        compression_fraction=fraction,
        groups=((0,), (1,), (2, 3)),
        seed=123,
    )


def _score(layer: int, method: str, fraction: float, value: float, suffix: str = ""):
    return LayerCandidateScoreContract(
        layer=layer,
        plan=_plan(layer, method, fraction, suffix),
        relative_perplexity_degradation=value,
        robust_score=value,
    )


def test_representative_prefixes_are_deterministic_nested_and_cover_layers() -> None:
    layers = tuple(range(12))
    order = representative_layer_order(layers)
    assert order[:3] == (5, 0, 11)
    assert set(order) == set(layers)
    assert resolve_sensitivity_layers(layers) == order[:5]
    prefixes = cumulative_layer_prefixes(layers, (1, 3, 6), include_all=True)
    assert tuple(len(prefix) for prefix in prefixes) == (1, 3, 6, 12)
    assert all(set(first) <= set(second) for first, second in zip(prefixes, prefixes[1:]))


def test_layer_winner_selection_prefers_robust_score_then_candidate_id() -> None:
    scores = (
        _score(0, "functional_cosine", 0.1, 0.03, "-b"),
        _score(0, "functional_cosine", 0.1, 0.02, "-a"),
    )
    winners = select_layer_winners(scores)
    assert tuple(item.plan.candidate_id for item in winners) == (
        "l0-functional_cosine-f0.10-a",
    )


def test_unmeasured_layers_use_nearest_sensitivity_score() -> None:
    plans = {
        layer: (_plan(layer, "functional_cosine", 0.1),)
        for layer in range(5)
    }
    scored = score_layer_plans(
        plans_by_layer=plans,
        measured_scores=(
            _score(0, "functional_cosine", 0.1, 0.01),
            _score(4, "functional_cosine", 0.1, 0.04),
        ),
    )
    middle = next(item for item in scored if item.layer == 2)
    assert middle.robust_score == 0.01
    assert middle.score_source_layer == 0
    assert not middle.measured


def test_unmeasured_layers_preserve_candidate_variant_scores() -> None:
    plans = {
        0: (
            _plan(0, "rmt_functional_usage", 0.1, "-a"),
            _plan(0, "rmt_functional_usage", 0.1, "-b"),
        ),
        1: (
            _plan(1, "rmt_functional_usage", 0.1, "-a"),
            _plan(1, "rmt_functional_usage", 0.1, "-b"),
        ),
    }
    scored = score_layer_plans(
        plans_by_layer=plans,
        measured_scores=(
            _score(0, "rmt_functional_usage", 0.1, 0.01, "-a"),
            _score(0, "rmt_functional_usage", 0.1, 0.04, "-b"),
        ),
    )
    layer_one = {item.plan.candidate_id: item.robust_score for item in scored if item.layer == 1}
    assert layer_one == {
        "l1-rmt_functional_usage-f0.10-a": 0.01,
        "l1-rmt_functional_usage-f0.10-b": 0.04,
    }


def test_cumulative_recipes_have_static_and_adaptive_variants() -> None:
    scores = tuple(
        _score(layer, "rmt_functional_usage", 0.25, 0.005)
        for layer in range(3)
    )
    recipes = build_cumulative_recipes(
        layer_prefixes=((1,), (1, 0, 2)),
        scored_plans=scores,
        methods=("rmt_functional_usage",),
        compression_fractions=(0.25,),
        refresh_modes=("static", "adaptive"),
    )
    assert len(recipes) == 4
    assert {recipe.trace_refresh_mode.value for recipe in recipes} == {
        "static",
        "adaptive",
    }
    assert {len(recipe.choices) for recipe in recipes} == {1, 3}


def test_global_allocator_uses_keep_and_respects_quality_constraints() -> None:
    scores = tuple(
        score
        for layer, costs in enumerate(((0.001, 0.02), (0.002, 0.03), (0.01, 0.04)))
        for fraction, cost in zip((0.1, 0.25), costs)
        for score in (_score(layer, "functional_cosine", fraction, cost),)
    )
    recipe = allocate_global_budget(
        target_compression=0.05,
        scored_plans=scores,
        all_layers=(0, 1, 2),
        quality_budget=0.01,
        max_layer_degradation=0.03,
        refresh_mode="static",
    )
    assert recipe.achieved_compression >= 0.05
    assert recipe.predicted_degradation <= 0.01
    assert all(choice.predicted_degradation <= 0.03 for choice in recipe.choices)
    assert len(recipe.choices) < 3


def test_next_step_advances_only_after_bounded_work_is_terminal() -> None:
    assert next_multilayer_step(
        (MultiLayerStep.TRACES_CALIBRATION, MultiLayerStep.TRACES_VALIDATION),
        remaining_diagnostic_layers=2,
        remaining_sensitivity_candidates=0,
        remaining_validation_recipes=0,
        remaining_test_recipes=0,
        remaining_materializations=0,
        remaining_materialized_benchmarks=0,
    ) == MultiLayerStep.LAYER_DIAGNOSTICS

    assert next_multilayer_step(
        (MultiLayerStep.SELECT_CONFIGURATIONS,),
        remaining_diagnostic_layers=0,
        remaining_sensitivity_candidates=0,
        remaining_validation_recipes=0,
        remaining_test_recipes=0,
        remaining_materializations=0,
        remaining_materialized_benchmarks=0,
        selection_written=True,
        selected_recipe_count=0,
    ) == MultiLayerStep.BUILD_SUMMARY
