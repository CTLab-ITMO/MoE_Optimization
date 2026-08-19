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
    MultiLayerValidationScoreContract,
    ProgressiveStageContract,
    ProgressiveValidationPlanContract,
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
    skipped_validation = _record_ids(
        records,
        config_hash=config_hash,
        record_type="skipped_multilayer_candidate",
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
        skipped_validation_recipe_ids=skipped_validation,
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


def build_hybrid_cumulative_recipes(
    *,
    layer_prefixes: Sequence[Sequence[int]],
    scored_plans: Sequence[LayerCandidateScoreContract],
    methods: Sequence[str],
    compression_fractions: Sequence[float],
    refresh_modes: Sequence[str],
) -> tuple[MultiLayerRecipeContract, ...]:
    allowed_methods = {str(method) for method in methods}
    winners = tuple(
        score
        for score in select_layer_winners(scored_plans)
        if score.plan.method.value in allowed_methods
    )
    by_layer_fraction: dict[
        tuple[int, float], list[LayerCandidateScoreContract]
    ] = defaultdict(list)
    for score in winners:
        by_layer_fraction[
            (score.layer, round(score.plan.compression_fraction, 12))
        ].append(score)

    all_layers = indexed_layers(
        {
            (
                score.layer,
                score.plan.method.value,
                round(score.plan.compression_fraction, 12),
            ): score
            for score in winners
        }
    )
    recipes: list[MultiLayerRecipeContract] = []
    for layers in layer_prefixes:
        selected_layers = tuple(int(layer) for layer in layers)
        for fraction in compression_fractions:
            rounded_fraction = round(float(fraction), 12)
            choices = []
            for layer in selected_layers:
                candidates = by_layer_fraction.get((layer, rounded_fraction), ())
                if not candidates:
                    raise ValueError(
                        f"No hybrid candidate for layer={layer}, fraction={fraction}"
                    )
                winner = min(
                    candidates,
                    key=lambda score: (
                        score.robust_score,
                        score.relative_perplexity_degradation,
                        score.plan.candidate_id,
                    ),
                )
                choices.append(_choice_from_score(winner))
            choice_tuple = tuple(choices)
            for refresh in refresh_modes:
                recipes.append(
                    MultiLayerRecipeContract(
                        recipe_id=(
                            f"hybrid-{refresh}-f{float(fraction):.3f}"
                            f"-n{len(selected_layers)}"
                        ),
                        kind=MultiLayerRecipeKind.HYBRID,
                        trace_refresh_mode=TraceRefreshMode(refresh),
                        choices=choice_tuple,
                        target_compression=float(fraction),
                        achieved_compression=(
                            float(fraction) * len(choice_tuple) / len(all_layers)
                        ),
                        predicted_degradation=sum(
                            choice.predicted_degradation for choice in choice_tuple
                        ),
                    )
                )
    return tuple(recipes)


def select_multilayer_configurations(
    scores: Sequence[MultiLayerValidationScoreContract],
    *,
    quality_budget: float,
    strategy: str,
    per_method_per_cell: int,
    include_global_winners: bool,
    max_selected: int,
    balance_layer_counts: bool = False,
    balance_refresh_modes: bool = True,
) -> tuple[str, ...]:
    if strategy not in {"global_winners", "method_balanced"}:
        raise ValueError("unsupported multi-layer selection strategy")
    if per_method_per_cell < 1:
        raise ValueError("per_method_per_cell must be positive")
    if max_selected < 0:
        raise ValueError("max_selected must be non-negative")
    eligible = tuple(
        sorted(
            (score for score in scores if score.robust_score <= quality_budget),
            key=lambda score: (score.robust_score, score.recipe_id),
        )
    )

    global_winners: dict[
        tuple[str, float, str], MultiLayerValidationScoreContract
    ] = {}
    for score in eligible:
        key = (
            score.kind.value,
            round(score.achieved_compression, 6),
            score.trace_refresh_mode.value,
        )
        current = global_winners.get(key)
        if current is None or (score.robust_score, score.recipe_id) < (
            current.robust_score,
            current.recipe_id,
        ):
            global_winners[key] = score
    mandatory_ids: set[str] = set()
    if strategy == "global_winners":
        selected_scores = tuple(global_winners.values())
    else:
        balanced: dict[tuple[object, ...], list[MultiLayerValidationScoreContract]] = (
            defaultdict(list)
        )
        for score in eligible:
            if score.kind == MultiLayerRecipeKind.GLOBAL_BUDGET:
                continue
            family = (
                "hybrid"
                if score.kind == MultiLayerRecipeKind.HYBRID
                or len(score.methods) != 1
                else score.methods[0]
            )
            key: tuple[object, ...] = (
                family,
                score.kind.value,
                round(score.target_compression, 6),
            )
            if balance_layer_counts:
                key += (score.layer_count,)
            if balance_refresh_modes:
                key += (score.trace_refresh_mode.value,)
            balanced[key].append(score)
        mandatory = tuple(
            score
            for key in sorted(balanced)
            for score in sorted(
                balanced[key],
                key=lambda item: (item.robust_score, item.recipe_id),
            )[:per_method_per_cell]
        )
        mandatory_ids = {score.recipe_id for score in mandatory}
        additions = tuple(global_winners.values()) if include_global_winners else ()
        selected_scores = mandatory + additions

    deduplicated = {
        score.recipe_id: score
        for score in selected_scores
    }
    ordered = tuple(
        sorted(
            deduplicated.values(),
            key=lambda score: (score.robust_score, score.recipe_id),
        )
    )
    if max_selected and len(ordered) > max_selected:
        if strategy == "method_balanced":
            if len(mandatory_ids) > max_selected:
                raise ValueError(
                    "max_selected is too small for the method-balanced quota"
                )
            mandatory_ordered = tuple(
                score for score in ordered if score.recipe_id in mandatory_ids
            )
            optional_ordered = tuple(
                score for score in ordered if score.recipe_id not in mandatory_ids
            )
            ordered = mandatory_ordered + optional_ordered[
                : max_selected - len(mandatory_ordered)
            ]
            ordered = tuple(
                sorted(
                    ordered,
                    key=lambda score: (score.robust_score, score.recipe_id),
                )
            )
        else:
            ordered = ordered[:max_selected]
    return tuple(score.recipe_id for score in ordered)


def _recipe_method_family(recipe: MultiLayerRecipeContract) -> str:
    methods = tuple(
        sorted(
            {
                choice.plan.method.value
                for choice in recipe.choices
                if choice.plan is not None
            }
        )
    )
    if recipe.kind == MultiLayerRecipeKind.HYBRID or len(methods) != 1:
        return "hybrid"
    return methods[0]


def select_materialization_ids(
    selected_ids: Sequence[str],
    recipes: Sequence[MultiLayerRecipeContract],
    *,
    count: int,
    strategy: str,
) -> tuple[str, ...]:
    if count < 1:
        raise ValueError("materialization count must be positive")
    if strategy not in {"selected_order", "method_balanced"}:
        raise ValueError("unsupported materialization selection strategy")
    ordered_ids = tuple(dict.fromkeys(str(value) for value in selected_ids))
    if strategy == "selected_order":
        return ordered_ids[:count]

    recipe_by_id = {recipe.recipe_id: recipe for recipe in recipes}
    unknown = tuple(recipe_id for recipe_id in ordered_ids if recipe_id not in recipe_by_id)
    if unknown:
        raise ValueError(f"Unknown selected recipe ids: {list(unknown)}")

    by_family: dict[str, list[tuple[int, str, MultiLayerRecipeContract]]] = defaultdict(list)
    for index, recipe_id in enumerate(ordered_ids):
        recipe = recipe_by_id[recipe_id]
        by_family[_recipe_method_family(recipe)].append((index, recipe_id, recipe))

    balanced: list[str] = []
    for family in sorted(by_family, key=lambda value: by_family[value][0][0]):
        _, recipe_id, _ = max(
            by_family[family],
            key=lambda item: (
                len(item[2].choices),
                item[2].achieved_compression,
                -item[0],
            ),
        )
        balanced.append(recipe_id)
    balanced.extend(recipe_id for recipe_id in ordered_ids if recipe_id not in balanced)
    return tuple(balanced[:count])


def plan_progressive_validation(
    recipes: Sequence[MultiLayerRecipeContract],
    scores: Sequence[MultiLayerValidationScoreContract],
    *,
    completed_ids: Sequence[str],
    failed_ids: Sequence[str],
    skipped_ids: Sequence[str],
    layer_counts: Sequence[int],
    quality_budget: float,
    min_accepted_recipes: int,
    stop_on_failure: bool,
) -> ProgressiveValidationPlanContract:
    configured_counts = tuple(sorted(set(int(value) for value in layer_counts)))
    if not configured_counts:
        raise ValueError("progressive layer counts must not be empty")
    if min_accepted_recipes < 1:
        raise ValueError("min_accepted_recipes must be positive")
    terminal = set(completed_ids) | set(failed_ids) | set(skipped_ids)
    score_by_id = {score.recipe_id: score for score in scores}
    progressive = tuple(
        recipe
        for recipe in recipes
        if recipe.kind in {
            MultiLayerRecipeKind.CUMULATIVE,
            MultiLayerRecipeKind.HYBRID,
        }
        and len(recipe.choices) in configured_counts
    )
    stages: list[ProgressiveStageContract] = []
    for layer_count in configured_counts:
        stage_ids = tuple(
            recipe.recipe_id
            for recipe in progressive
            if len(recipe.choices) == layer_count
        )
        pending = tuple(recipe_id for recipe_id in stage_ids if recipe_id not in terminal)
        completed = tuple(recipe_id for recipe_id in stage_ids if recipe_id in completed_ids)
        accepted = tuple(
            recipe_id
            for recipe_id in completed
            if recipe_id in score_by_id
            and score_by_id[recipe_id].robust_score <= quality_budget
        )
        passed = not pending and len(accepted) >= min_accepted_recipes
        stages.append(
            ProgressiveStageContract(
                layer_count=layer_count,
                recipe_count=len(stage_ids),
                completed_count=len(completed),
                accepted_count=len(accepted),
                passed=passed,
            )
        )
        if pending:
            return ProgressiveValidationPlanContract(
                current_layer_count=layer_count,
                pending_recipe_ids=pending,
                skipped_recipe_ids=(),
                stages=tuple(stages),
            )
        if not passed and stop_on_failure:
            later = tuple(
                recipe.recipe_id
                for recipe in recipes
                if recipe.recipe_id not in terminal
                and recipe.recipe_id not in stage_ids
            )
            return ProgressiveValidationPlanContract(
                current_layer_count=None,
                pending_recipe_ids=(),
                skipped_recipe_ids=later,
                stages=tuple(stages),
            )

    progressive_ids = {recipe.recipe_id for recipe in progressive}
    remaining = tuple(
        recipe.recipe_id
        for recipe in recipes
        if recipe.recipe_id not in progressive_ids
        and recipe.recipe_id not in terminal
    )
    return ProgressiveValidationPlanContract(
        current_layer_count=None,
        pending_recipe_ids=remaining,
        skipped_recipe_ids=(),
        stages=tuple(stages),
    )


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
