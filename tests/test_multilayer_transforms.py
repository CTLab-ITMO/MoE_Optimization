from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from common.moe_utils import MoeHandle
from moe_opt_core.adapters import HFModelAdapter
from moe_opt_core.experiment import (
    CandidateAction,
    CandidateMethod,
    CandidatePlanContract,
    LayerCompressionChoiceContract,
    MoeLayerRuntimeContract,
    MultiLayerRecipeContract,
    MultiLayerRecipeKind,
    TraceRefreshMode,
)
from moe_opt_runner.materialization import (
    load_materialized_recipe_bundle,
    materialize_recipe_bundle,
)
from moe_opt_runner.runtime import RuntimeContext
from moe_opt_runner.transforms import (
    apply_multi_layer_transform_inplace,
    reversible_multi_layer_transform,
)


class TinyMoe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate = nn.Linear(4, 4, bias=False)
        self.experts = nn.ModuleList([nn.Linear(4, 4) for _ in range(4)])
        self.num_experts = 4
        self.top_k = 2


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([TinyMoe(), TinyMoe()])


def _runtime() -> RuntimeContext:
    model = TinyModel()
    handles = {
        layer: MoeHandle(
            layer_idx=layer,
            moe=moe,
            gate=moe.gate,
            experts=moe.experts,
            kind="tiny",
        )
        for layer, moe in enumerate(model.layers)
    }
    contracts = {
        layer: MoeLayerRuntimeContract(
            layer=layer,
            kind="tiny",
            routed_experts=4,
            shared_experts=0,
            configured_routed_top_k=2,
            requested_routed_top_k=2,
            effective_routed_top_k=2,
            top_k_updated=False,
        )
        for layer in handles
    }
    return RuntimeContext(
        model=model,
        adapter=HFModelAdapter(model=model),
        handles_by_layer=handles,
        runtime_contracts_by_layer=contracts,
    )


def _plan(layer: int) -> CandidatePlanContract:
    return CandidatePlanContract(
        candidate_id=f"merge-layer-{layer}",
        layer=layer,
        method=CandidateMethod.FUNCTIONAL_COSINE,
        action=CandidateAction.MERGE,
        compression_fraction=0.25,
        groups=((0, 1), (2,), (3,)),
        seed=1,
    )


def _recipe() -> MultiLayerRecipeContract:
    choices = tuple(
        LayerCompressionChoiceContract(
            layer=layer,
            plan=_plan(layer),
            compression_fraction=0.25,
            predicted_degradation=0.01,
            score_source_layer=layer,
        )
        for layer in (0, 1)
    )
    return MultiLayerRecipeContract(
        recipe_id="two-layer",
        kind=MultiLayerRecipeKind.CUMULATIVE,
        trace_refresh_mode=TraceRefreshMode.STATIC,
        choices=choices,
        target_compression=0.25,
        achieved_compression=0.25,
        predicted_degradation=0.02,
    )


def test_reversible_multilayer_transform_restores_every_layer() -> None:
    ctx = _runtime()
    gates = tuple(ctx.handles_by_layer[layer].gate for layer in (0, 1))
    with reversible_multi_layer_transform(
        ctx,
        (_plan(0), _plan(1)),
        merge_method="baseline",
        geometry_rank=2,
        shrink_tau=0.0,
    ) as transforms:
        assert len(transforms) == 2
        assert tuple(len(ctx.handles_by_layer[layer].experts) for layer in (0, 1)) == (3, 3)
    assert tuple(ctx.handles_by_layer[layer].gate for layer in (0, 1)) == gates
    assert tuple(len(ctx.handles_by_layer[layer].experts) for layer in (0, 1)) == (4, 4)


def test_inplace_multilayer_transform_changes_all_layers_without_snapshots() -> None:
    ctx = _runtime()
    apply_multi_layer_transform_inplace(
        ctx,
        (_plan(0), _plan(1)),
        merge_method="baseline",
        geometry_rank=2,
        shrink_tau=0.0,
    )
    assert tuple(len(ctx.handles_by_layer[layer].experts) for layer in (0, 1)) == (3, 3)
    assert tuple(ctx.model.layers[layer].num_experts for layer in (0, 1)) == (3, 3)


def test_materialized_bundle_round_trip_recreates_structure_and_weights(
    tmp_path: Path,
) -> None:
    source = _runtime()
    bundle = materialize_recipe_bundle(
        source,
        _recipe(),
        tmp_path,
        config_hash="hash",
        merge_method="baseline",
        geometry_rank=2,
        shrink_tau=0.0,
        save_layer_weights=True,
    )
    target = _runtime()
    loaded = load_materialized_recipe_bundle(
        target,
        bundle,
        expected_config_hash="hash",
        merge_method="baseline",
        geometry_rank=2,
        shrink_tau=0.0,
    )
    assert loaded == _recipe()
    for layer in (0, 1):
        assert len(target.handles_by_layer[layer].experts) == 3
        source_state = source.handles_by_layer[layer].moe.state_dict()
        target_state = target.handles_by_layer[layer].moe.state_dict()
        assert source_state.keys() == target_state.keys()
        assert all(torch.equal(source_state[key], target_state[key]) for key in source_state)
