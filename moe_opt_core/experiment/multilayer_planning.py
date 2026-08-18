from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import math
from typing import Iterable, Mapping, Sequence

from moe_opt_core.experiment.contracts import (
    CandidatePlanContract,
    LayerCandidateScoreContract,
    LayerCompressionChoiceContract,
    MultiLayerRecipeContract,
    MultiLayerRecipeKind,
    MultiLayerExecutionStateContract,
    MultiLayerStep,
    TraceRefreshMode,
)


def _record_ids(
    records: Iterable[Mapping[str, object]],
    *,
    config_hash: str,
    record_type: str,
    id_field: str,
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                str(row[id_field])
                for row in records
                if row.get("config_hash") == config_hash
                and row.get("record_type") == record_type
                and row.get(id_field) is not None
            }
        )
    )


def build_multilayer_execution_state(
    *,
    config_hash: str,
    records: Sequence[Mapping[str, object]],
    artifact_steps: Sequence[MultiLayerStep] = (),
    selected_recipe_ids: Sequence[str] = (),
) -> MultiLayerExecutionStateContract:
    completed = set(artifact_steps)
    completed_sensitivity = _record_ids(
        records,
        config_hash=config_hash,
        record_type="sensitivity_candidate",
        id_field="candidate_id",
    )
    failed_sensitivity = _record_ids(
        records,
        config_hash=config_hash,
        record_type="failed_sensitivity_candidate",
        id_field="candidate_id",
    )
    completed_validation = _record_ids(
        records,
        config_hash=config_hash,
        record_type="multilayer_candidate",
        id_field="recipe_id",
    )
    failed_validation = _record_ids(
        records,
        config_hash=config_hash,
        record_type="failed_multilayer_candidate",
        id_field="recipe_id",
    )
    completed_test = _record_ids(
        records,
        config_hash=config_hash,
        record_type="selected_multilayer_candidate",
        id_field="recipe_id",
    )
    failed_test = _record_ids(
        records,
        config_hash=config_hash,
        record_type="failed_selected_multilayer_candidate",
        id_field="recipe_id",
    )
    return MultiLayerExecutionStateContract(
        schema_version="1.0",
        config_hash=config_hash,
        completed_steps=tuple(step for step in MultiLayerStep if step in completed),
        completed_sensitivity_ids=completed_sensitivity,
        failed_sensitivity_ids=failed_sensitivity,
        completed_validation_recipe_ids=completed_validation,
        failed_validation_recipe_ids=failed_validation,
        selected_recipe_ids=tuple(selected_recipe_ids),
        completed_test_recipe_ids=completed_test,
        failed_test_recipe_ids=failed_test,
    )


def pending_multilayer_ids(
    all_ids: Sequence[str],
    *,
    completed_ids: Sequence[str],
    failed_ids: Sequence[str],
    retry_failed: bool,
) -> tuple[str, ...]:
    terminal = set(completed_ids)
    if not retry_failed:
        terminal.update(failed_ids)
    return tuple(item_id for item_id in all_ids if item_id not in terminal)


def next_multilayer_step(
    completed_steps: Sequence[MultiLayerStep],
    *,
    remaining_diagnostic_layers: int,
    remaining_sensitivity_candidates: int,
    remaining_validation_recipes: int,
    remaining_test_recipes: int,
    remaining_materializations: int,
    remaining_materialized_benchmarks: int,
    selection_written: bool = False,
    selected_recipe_count: int = 0,
) -> MultiLayerStep | None:
    completed = set(completed_steps)
    if selection_written and selected_recipe_count == 0:
        return (
            None
            if MultiLayerStep.BUILD_SUMMARY in completed
            else MultiLayerStep.BUILD_SUMMARY
        )
    ordered_checks = (
        (MultiLayerStep.TRACES_CALIBRATION, None),
        (MultiLayerStep.TRACES_VALIDATION, None),
        (MultiLayerStep.LAYER_DIAGNOSTICS, remaining_diagnostic_layers),
        (MultiLayerStep.BASELINE_VALIDATION, None),
        (MultiLayerStep.SENSITIVITY_VALIDATION, remaining_sensitivity_candidates),
        (MultiLayerStep.BUILD_RECIPES, None),
        (MultiLayerStep.MULTILAYER_VALIDATION, remaining_validation_recipes),
        (MultiLayerStep.SELECT_CONFIGURATIONS, None),
        (MultiLayerStep.TRACES_TEST, None),
        (MultiLayerStep.BASELINE_TEST, None),
        (MultiLayerStep.TEST_SELECTED, remaining_test_recipes),
        (MultiLayerStep.MATERIALIZE, remaining_materializations),
        (
            MultiLayerStep.BENCHMARK_MATERIALIZED,
            remaining_materialized_benchmarks,
        ),
        (MultiLayerStep.BUILD_SUMMARY, None),
    )
    for step, remaining in ordered_checks:
        if remaining is not None:
            if remaining > 0:
                return step
            continue
        if step not in completed:
            return step
    return None


def representative_layer_order(layers: Sequence[int]) -> tuple[int, ...]:
    """Return deterministic nested prefixes spanning middle, ends, then gaps."""
    ordered = tuple(sorted(set(int(layer) for layer in layers)))
    if not ordered:
        raise ValueError("layers must not be empty")
    middle = ordered[(len(ordered) - 1) // 2]
    selected = [middle]
    remaining = set(ordered) - {middle}
    for endpoint in (ordered[0], ordered[-1]):
        if endpoint in remaining:
            selected.append(endpoint)
            remaining.remove(endpoint)
    while remaining:
        candidate = max(
            remaining,
            key=lambda layer: (
                min(abs(layer - chosen) for chosen in selected),
                -layer,
            ),
        )
        selected.append(candidate)
        remaining.remove(candidate)
    return tuple(selected)


def resolve_sensitivity_layers(
    configured_layers: Sequence[int],
    explicit_layers: Sequence[int] = (),
    *,
    default_count: int = 5,
) -> tuple[int, ...]:
    configured = tuple(sorted(set(int(layer) for layer in configured_layers)))
    if not configured:
        raise ValueError("configured_layers must not be empty")
    explicit = tuple(int(layer) for layer in explicit_layers)
    if explicit:
        unknown = sorted(set(explicit) - set(configured))
        if unknown:
            raise ValueError(f"Unknown sensitivity layers: {unknown}")
        return explicit
    return representative_layer_order(configured)[: min(default_count, len(configured))]


def cumulative_layer_prefixes(
    layers: Sequence[int],
    counts: Sequence[int],
    *,
    include_all: bool,
) -> tuple[tuple[int, ...], ...]:
    priority = representative_layer_order(layers)
    requested = {int(count) for count in counts}
    if include_all:
        requested.add(len(priority))
    invalid = sorted(count for count in requested if count < 1 or count > len(priority))
    if invalid:
        raise ValueError(f"Invalid cumulative layer counts: {invalid}")
    return tuple(priority[:count] for count in sorted(requested))


def select_layer_winners(
    scores: Iterable[LayerCandidateScoreContract],
) -> tuple[LayerCandidateScoreContract, ...]:
    winners: dict[tuple[int, str, float], LayerCandidateScoreContract] = {}
    for score in scores:
        key = (
            score.layer,
            score.plan.method.value,
            round(score.plan.compression_fraction, 12),
        )
        current = winners.get(key)
        rank = (
            score.robust_score,
            score.relative_perplexity_degradation,
            score.plan.candidate_id,
        )
        if current is None or rank < (
            current.robust_score,
            current.relative_perplexity_degradation,
            current.plan.candidate_id,
        ):
            winners[key] = score
    return tuple(winners[key] for key in sorted(winners))


def _nearest_score(
    *,
    layer: int,
    plan: CandidatePlanContract,
    scores: Sequence[LayerCandidateScoreContract],
) -> LayerCandidateScoreContract:
    variant_key = _plan_variant_key(plan)
    matching = tuple(
        score
        for score in scores
        if score.plan.method == plan.method
        and math.isclose(
            score.plan.compression_fraction,
            plan.compression_fraction,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and _plan_variant_key(score.plan) == variant_key
    )
    if not matching:
        matching = tuple(
            score
            for score in scores
            if score.plan.method == plan.method
            and math.isclose(
                score.plan.compression_fraction,
                plan.compression_fraction,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
    if not matching:
        raise ValueError(
            "No sensitivity score for "
            f"method={plan.method.value}, fraction={plan.compression_fraction}"
        )
    return min(matching, key=lambda score: (abs(score.layer - layer), score.layer))


def _plan_variant_key(plan: CandidatePlanContract) -> str:
    prefix = f"l{plan.layer}-"
    return (
        plan.candidate_id[len(prefix) :]
        if plan.candidate_id.startswith(prefix)
        else plan.candidate_id
    )


def score_layer_plans(
    *,
    plans_by_layer: Mapping[int, Sequence[CandidatePlanContract]],
    measured_scores: Sequence[LayerCandidateScoreContract],
) -> tuple[LayerCandidateScoreContract, ...]:
    result: list[LayerCandidateScoreContract] = []
    for layer in sorted(plans_by_layer):
        for plan in plans_by_layer[layer]:
            source = _nearest_score(
                layer=layer,
                plan=plan,
                scores=measured_scores,
            )
            result.append(
                LayerCandidateScoreContract(
                    layer=layer,
                    plan=plan,
                    relative_perplexity_degradation=(
                        source.relative_perplexity_degradation
                    ),
                    robust_score=source.robust_score,
                    measured=source.layer == layer,
                    score_source_layer=source.layer,
                )
            )
    return tuple(result)


def _choice_from_score(
    score: LayerCandidateScoreContract,
) -> LayerCompressionChoiceContract:
    return LayerCompressionChoiceContract(
        layer=score.layer,
        plan=score.plan,
        compression_fraction=score.plan.compression_fraction,
        predicted_degradation=max(0.0, score.robust_score),
        score_source_layer=(
            score.layer
            if score.score_source_layer is None
            else score.score_source_layer
        ),
    )


def build_cumulative_recipes(
    *,
    layer_prefixes: Sequence[Sequence[int]],
    scored_plans: Sequence[LayerCandidateScoreContract],
    methods: Sequence[str],
    compression_fractions: Sequence[float],
    refresh_modes: Sequence[str],
) -> tuple[MultiLayerRecipeContract, ...]:
    indexed: dict[tuple[int, str, float], LayerCandidateScoreContract] = {}
    for score in select_layer_winners(scored_plans):
        indexed[
            (
                score.layer,
                score.plan.method.value,
                round(score.plan.compression_fraction, 12),
            )
        ] = score
    recipes: list[MultiLayerRecipeContract] = []
    for layers in layer_prefixes:
        selected_layers = tuple(int(layer) for layer in layers)
        for method in methods:
            for fraction in compression_fractions:
                choices = tuple(
                    _choice_from_score(
                        indexed[(layer, str(method), round(float(fraction), 12))]
                    )
                    for layer in selected_layers
                )
                for refresh in refresh_modes:
                    predicted = sum(choice.predicted_degradation for choice in choices)
                    recipe_id = (
                        f"cumulative-{refresh}-{method}-f{float(fraction):.3f}"
                        f"-n{len(selected_layers)}"
                    )
                    recipes.append(
                        MultiLayerRecipeContract(
                            recipe_id=recipe_id,
                            kind=MultiLayerRecipeKind.CUMULATIVE,
                            trace_refresh_mode=TraceRefreshMode(refresh),
                            choices=choices,
                            target_compression=float(fraction),
                            achieved_compression=(
                                float(fraction) * len(choices) / len(indexed_layers(indexed))
                            ),
                            predicted_degradation=predicted,
                        )
                    )
    return tuple(recipes)


def indexed_layers(
    indexed: Mapping[tuple[int, str, float], LayerCandidateScoreContract],
) -> tuple[int, ...]:
    return tuple(sorted({key[0] for key in indexed}))


def allocate_global_budget(
    *,
    target_compression: float,
    scored_plans: Sequence[LayerCandidateScoreContract],
    all_layers: Sequence[int],
    quality_budget: float,
    max_layer_degradation: float,
    refresh_mode: str,
) -> MultiLayerRecipeContract:
    layers = tuple(sorted(set(int(layer) for layer in all_layers)))
    if not layers:
        raise ValueError("all_layers must not be empty")
    if not 0.0 < target_compression < 1.0:
        raise ValueError("target_compression must be in (0, 1)")

    by_layer: dict[int, list[LayerCandidateScoreContract]] = defaultdict(list)
    for score in select_layer_winners(scored_plans):
        if max(0.0, score.robust_score) <= max_layer_degradation:
            by_layer[score.layer].append(score)

    # State: removed basis points -> (quality cost, choices).
    states: dict[int, tuple[float, tuple[LayerCompressionChoiceContract, ...]]] = {
        0: (0.0, ())
    }
    for layer in layers:
        options: list[LayerCompressionChoiceContract] = [
            LayerCompressionChoiceContract(
                layer=layer,
                plan=None,
                compression_fraction=0.0,
                predicted_degradation=0.0,
                score_source_layer=layer,
            )
        ]
        options.extend(_choice_from_score(score) for score in by_layer.get(layer, ()))
        updated: dict[int, tuple[float, tuple[LayerCompressionChoiceContract, ...]]] = {}
        for removed, (cost, choices) in states.items():
            for option in options:
                new_cost = cost + option.predicted_degradation
                if new_cost > quality_budget + 1e-12:
                    continue
                new_removed = removed + round(option.compression_fraction * 10_000)
                value = (new_cost, choices + (option,))
                current = updated.get(new_removed)
                if current is None or (new_cost, _choice_ids(value[1])) < (
                    current[0],
                    _choice_ids(current[1]),
                ):
                    updated[new_removed] = value
        states = updated
    required = math.ceil(target_compression * len(layers) * 10_000 - 1e-12)
    feasible = tuple(
        (removed, cost, choices)
        for removed, (cost, choices) in states.items()
        if removed >= required
    )
    if not feasible:
        raise ValueError(
            "No global allocation satisfies compression and quality constraints"
        )
    removed, cost, choices = min(
        feasible,
        key=lambda value: (value[1], value[0] - required, _choice_ids(value[2])),
    )
    active = tuple(choice for choice in choices if choice.plan is not None)
    return MultiLayerRecipeContract(
        recipe_id=f"global-{refresh_mode}-t{target_compression:.3f}",
        kind=MultiLayerRecipeKind.GLOBAL_BUDGET,
        trace_refresh_mode=TraceRefreshMode(refresh_mode),
        choices=active,
        target_compression=float(target_compression),
        achieved_compression=removed / (len(layers) * 10_000.0),
        predicted_degradation=cost,
    )


def _choice_ids(choices: Sequence[LayerCompressionChoiceContract]) -> tuple[str, ...]:
    return tuple(
        choice.plan.candidate_id if choice.plan is not None else f"keep-{choice.layer}"
        for choice in choices
    )


def replace_recipe_plans(
    recipe: MultiLayerRecipeContract,
    plans: Mapping[int, CandidatePlanContract],
) -> MultiLayerRecipeContract:
    """Replace static plans with plans recomputed from adaptive traces."""
    choices = tuple(
        replace(choice, plan=plans.get(choice.layer, choice.plan))
        for choice in recipe.choices
    )
    return replace(recipe, choices=choices)
