from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Mapping

import torch

from moe_opt_core.experiment.contracts import MultiLayerRecipeContract
from moe_opt_core.experiment.serialization import (
    multilayer_recipe_from_dict,
    multilayer_recipe_to_dict,
)
from moe_opt_runner.pilot_artifacts import write_json_atomic
from moe_opt_runner.runtime import RuntimeContext
from moe_opt_runner.transforms import (
    apply_multi_layer_transform_inplace,
    model_parameter_mb,
)


def materialize_recipe_bundle(
    ctx: RuntimeContext,
    recipe: MultiLayerRecipeContract,
    output_dir: str | Path,
    *,
    config_hash: str,
    merge_method: str,
    geometry_rank: int,
    shrink_tau: float,
    save_layer_weights: bool,
) -> Path:
    destination = Path(output_dir) / recipe.recipe_id
    destination.mkdir(parents=True, exist_ok=True)
    plans = tuple(
        choice.plan for choice in recipe.choices if choice.plan is not None
    )
    transforms = apply_multi_layer_transform_inplace(
        ctx,
        plans,
        merge_method=merge_method,
        geometry_rank=geometry_rank,
        shrink_tau=shrink_tau,
    )
    layer_artifacts: dict[str, str] = {}
    if save_layer_weights:
        for plan in plans:
            path = destination / f"layer_{plan.layer}.pt"
            torch.save(ctx.handles_by_layer[plan.layer].moe.state_dict(), path)
            layer_artifacts[str(plan.layer)] = path.name
    manifest = {
        "schema_version": "1.0",
        "config_hash": config_hash,
        "recipe": multilayer_recipe_to_dict(recipe),
        "transforms": [asdict(item) for item in transforms],
        "model_parameter_mb": model_parameter_mb(ctx.model),
        "layer_artifacts": layer_artifacts,
        "load_peak_note": (
            "The bundle is applied to a base model before layer weights are loaded; "
            "initial base-model load memory is not eliminated."
        ),
    }
    write_json_atomic(destination / "manifest.json", manifest)
    return destination


def load_materialized_recipe_bundle(
    ctx: RuntimeContext,
    bundle_dir: str | Path,
    *,
    expected_config_hash: str,
    merge_method: str,
    geometry_rank: int,
    shrink_tau: float,
) -> MultiLayerRecipeContract:
    root = Path(bundle_dir)
    payload = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if payload.get("config_hash") != expected_config_hash:
        raise ValueError("Materialized bundle belongs to another experiment config")
    raw_recipe = payload.get("recipe")
    if not isinstance(raw_recipe, Mapping):
        raise ValueError("Materialized bundle has no recipe")
    recipe = multilayer_recipe_from_dict(raw_recipe)
    plans = tuple(
        choice.plan for choice in recipe.choices if choice.plan is not None
    )
    apply_multi_layer_transform_inplace(
        ctx,
        plans,
        merge_method=merge_method,
        geometry_rank=geometry_rank,
        shrink_tau=shrink_tau,
    )
    layer_artifacts = payload.get("layer_artifacts", {})
    if isinstance(layer_artifacts, Mapping):
        for plan in plans:
            filename = layer_artifacts.get(str(plan.layer))
            if filename:
                state = torch.load(root / str(filename), map_location="cpu")
                ctx.handles_by_layer[plan.layer].moe.load_state_dict(state)
    return recipe
