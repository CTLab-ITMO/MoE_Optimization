from __future__ import annotations

from typing import Mapping

from moe_opt_core.experiment.contracts import (
    CandidateAction,
    CandidateMethod,
    CandidatePlanContract,
    LayerCompressionChoiceContract,
    MergeWeightingPolicy,
    MultiLayerRecipeContract,
    MultiLayerRecipeKind,
    TraceRefreshMode,
)


def candidate_plan_to_dict(plan: CandidatePlanContract) -> dict[str, object]:
    return {
        "candidate_id": plan.candidate_id,
        "layer": plan.layer,
        "method": plan.method.value,
        "action": plan.action.value,
        "compression_fraction": plan.compression_fraction,
        "groups": [list(group) for group in plan.groups],
        "seed": plan.seed,
        "diagnostics": dict(plan.diagnostics),
        "merge_weighting": plan.merge_weighting.value,
        "group_merge_weights": [list(weights) for weights in plan.group_merge_weights],
    }


def candidate_plan_from_dict(payload: Mapping[str, object]) -> CandidatePlanContract:
    diagnostics = payload.get("diagnostics", {})
    return CandidatePlanContract(
        candidate_id=str(payload["candidate_id"]),
        layer=int(payload["layer"]),
        method=CandidateMethod(str(payload["method"])),
        action=CandidateAction(str(payload["action"])),
        compression_fraction=float(payload["compression_fraction"]),
        groups=tuple(
            tuple(int(index) for index in group)
            for group in payload.get("groups", ())
        ),
        seed=int(payload["seed"]),
        diagnostics=tuple(
            (str(key), float(value))
            for key, value in (
                diagnostics.items() if isinstance(diagnostics, Mapping) else ()
            )
        ),
        merge_weighting=MergeWeightingPolicy(
            str(payload.get("merge_weighting", "uniform"))
        ),
        group_merge_weights=tuple(
            tuple(float(value) for value in weights)
            for weights in payload.get("group_merge_weights", ())
        ),
    )


def multilayer_recipe_to_dict(recipe: MultiLayerRecipeContract) -> dict[str, object]:
    return {
        "recipe_id": recipe.recipe_id,
        "kind": recipe.kind.value,
        "trace_refresh_mode": recipe.trace_refresh_mode.value,
        "target_compression": recipe.target_compression,
        "achieved_compression": recipe.achieved_compression,
        "predicted_degradation": recipe.predicted_degradation,
        "choices": [
            {
                "layer": choice.layer,
                "plan": (
                    candidate_plan_to_dict(choice.plan)
                    if choice.plan is not None
                    else None
                ),
                "compression_fraction": choice.compression_fraction,
                "predicted_degradation": choice.predicted_degradation,
                "score_source_layer": choice.score_source_layer,
            }
            for choice in recipe.choices
        ],
    }


def multilayer_recipe_from_dict(
    payload: Mapping[str, object],
) -> MultiLayerRecipeContract:
    choices = []
    for raw_choice in payload.get("choices", ()):
        if not isinstance(raw_choice, Mapping):
            raise ValueError("Recipe choices must be objects")
        raw_plan = raw_choice.get("plan")
        choices.append(
            LayerCompressionChoiceContract(
                layer=int(raw_choice["layer"]),
                plan=(
                    candidate_plan_from_dict(raw_plan)
                    if isinstance(raw_plan, Mapping)
                    else None
                ),
                compression_fraction=float(raw_choice["compression_fraction"]),
                predicted_degradation=float(raw_choice["predicted_degradation"]),
                score_source_layer=int(raw_choice["score_source_layer"]),
            )
        )
    return MultiLayerRecipeContract(
        recipe_id=str(payload["recipe_id"]),
        kind=MultiLayerRecipeKind(str(payload["kind"])),
        trace_refresh_mode=TraceRefreshMode(str(payload["trace_refresh_mode"])),
        choices=tuple(choices),
        target_compression=float(payload["target_compression"]),
        achieved_compression=float(payload["achieved_compression"]),
        predicted_degradation=float(payload["predicted_degradation"]),
    )
