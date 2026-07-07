"""Stage 3 -- fusion.

Merges expert groups for every layer using all registered fusion methods,
scores each result on validation data, and writes one versioned FusedExpert
per (method, layer, group) plus Best_<metric> winners.

Idempotent: each artifact stores an upstream hash; re-runs skip groups whose
outputs are already fresh.
"""
import os

import torch

from common import layout, provenance
from common.artifacts import GroupActivationIndex, Partition, SavedExpert
from common.context import StageContext
from common.fused_group import FusedExpert
from common.fusion import FUSION_REGISTRY
from common.metrics import METRIC_REGISTRY, live_best_metrics
from common.mlp_utils import (
    FUSION_METHODS,
    select_best_fused,
    test_model_on_data,
)
from common.names import BEST_PREFIX, TIME_METRIC
from common.registry import require_all_registered
from common.seeding import set_seed
from common.serialization import make_provenance
from experts_fusion.config import (
    DATA_DIR,
    EXPERTS,
    FILTER_FUSIONS,
    FILTER_METRICS,
    LAYERS,
    MIN_PAIR_VALIDATION_SAMPLES,
    SEED,
    config_snapshot_dict,
    device,
)

# Check config filters against registries and verify no config drift since stage 1.
require_all_registered(FILTER_FUSIONS, FUSION_REGISTRY, field="FILTER_FUSIONS")
require_all_registered(FILTER_METRICS, METRIC_REGISTRY, field="FILTER_METRICS")
provenance.assert_config_matches_snapshot(DATA_DIR, config_snapshot_dict())
layout.validate_layout_version(DATA_DIR)

fusion_methods = dict(FUSION_METHODS)
if FILTER_FUSIONS is not None:
    fusion_methods = {k: v for k, v in fusion_methods.items() if k in FILTER_FUSIONS}
selected_best_metrics = live_best_metrics()
if FILTER_METRICS is not None:
    selected_best_metrics = {k: v for k, v in selected_best_metrics.items() if k in FILTER_METRICS}

set_seed(SEED)
# Per-layer/per-group copies give stochastic methods (e.g. Fisher) an order-independent RNG.
base_ctx = StageContext(seed=SEED, device=device)
config_fingerprint = provenance.config_hash(config_snapshot_dict())
# Idempotency (skip-if-done) hashes over the fusion-relevant config only: eval-only
# flags (REPLACE_*, INSPECT_FUSION_METHOD, USE_DEFAULT_EXPERTS) are consumed by stage 4
# and must not invalidate already-fused experts.
fusion_fingerprint = provenance.fusion_config_hash(config_snapshot_dict())


def _gather(mlp_inputs, indices):
    return [mlp_inputs[i].to(device) for i in indices]


def _save_fused(name, layer, group, mlp, upstream_hash) -> None:
    os.makedirs(layout.fusion_experts_dir(DATA_DIR, name, layer), exist_ok=True)
    FusedExpert(mlp).save(
        layout.fusion_expert_path(DATA_DIR, name, layer, group),
        provenance=make_provenance(
            seed=SEED, config_hash=config_fingerprint, upstream_hash=upstream_hash
        ),
    )


method_names = list(fusion_methods.keys())
best_names = [f"{BEST_PREFIX}{metric}" for metric in selected_best_metrics]

# Reconstruction reports for diagnostic plots. resumed_any tracks whether any group
# was skipped; if so, plotting is suppressed to avoid overwriting a complete plot
# with a partial one.
experiment_results = []
resumed_any = False

# Iterate the MoE layers stage 1 produced (run manifest), not the raw config
# LAYERS: dense layers skipped by stage 1 have no grouping/fusion inputs (CR#4).
# Falls back to LAYERS when no manifest is present (e.g. isolated stage runs).
produced_layers = provenance.resolve_produced_layers(DATA_DIR, LAYERS)

for layer in produced_layers:
    partition = Partition.load(layout.group_path(DATA_DIR, layer))
    partition_fingerprint = provenance.header_hash(layout.group_path(DATA_DIR, layer))
    train_index = GroupActivationIndex.load(
        layout.group_activations_path(DATA_DIR, layout.TRAIN_SPLIT, layer)
    ).to_lists()
    validation_index = GroupActivationIndex.load(
        layout.group_activations_path(DATA_DIR, layout.VALIDATION_SPLIT, layer)
    ).to_lists()
    experts = [
        SavedExpert.load(layout.expert_path(DATA_DIR, layer, i)).mlp
        for i in range(EXPERTS)
    ]
    train_mlp_inputs = torch.load(layout.mlp_inputs_path(DATA_DIR, layout.TRAIN_SPLIT, layer))
    validation_mlp_inputs = torch.load(
        layout.mlp_inputs_path(DATA_DIR, layout.VALIDATION_SPLIT, layer)
    )

    for group_idx, group in enumerate(partition.groups):
        def upstream(name):
            return provenance.fused_upstream_hash(
                config_hash=fusion_fingerprint,
                layer=layer,
                method=name,
                group=group_idx,
                extra=partition_fingerprint,
            )

        # Skip this group if all outputs are already fresh (Best_* needs all methods).
        if all(
            provenance.is_fresh(
                layout.fusion_expert_path(DATA_DIR, name, layer, group_idx), upstream(name)
            )
            for name in method_names + best_names
        ):
            resumed_any = True
            continue

        member_a, member_b = group.members  # k == 2 (our implementations)
        expert_a = experts[member_a].to(device)
        expert_b = experts[member_b].to(device)
        train_data = _gather(train_mlp_inputs, train_index[group_idx])
        validation_data = _gather(validation_mlp_inputs, validation_index[group_idx])
        ctx = base_ctx.for_layer(layer).for_group(group_idx)

        # Call each fusion method via the k-agnostic interface (positional mlps + keyword train_data).
        fused = {
            name: method(expert_a, expert_b, train_data=train_data, ctx=ctx)
            for name, method in fusion_methods.items()
        }
        results = None
        if len(validation_data) > MIN_PAIR_VALIDATION_SAMPLES:
            results = {
                name: test_model_on_data(fused[name], expert_a, expert_b, validation_data)
                for name in method_names
            }
        chosen = select_best_fused(
            fused, results, selected_best_metrics, method_names=method_names
        )
        if results is not None:
            experiment_results.append(results)

        for name, mlp in list(fused.items()) + list(chosen.items()):
            _save_fused(name, layer, group_idx, mlp, upstream(name))

    print(f"layer {layer}: fused {partition.num_groups} groups")


def _plot_fusion_diagnostics() -> None:
    """Plot per-method reconstruction metric distributions (best-effort).

    Outputs go to DATA_DIR/plots. Skipped on partial resumes to avoid
    overwriting complete plots with a subset. Failures are caught by the caller.
    """
    if resumed_any or not experiment_results:
        return
    from common import plots

    plot_metrics = list(selected_best_metrics.keys()) + [TIME_METRIC]
    data = {
        metric: {
            name: [report[name][metric] for report in experiment_results]
            for name in method_names
        }
        for metric in plot_metrics
    }
    medians = plots.metric_medians(data, plot_metrics, method_names)
    for metric in plot_metrics:
        print(f"median {metric}:")
        for name in method_names:
            print(f"  {name}: {medians[metric][name]:.3f}")
    out_dir = layout.plots_dir(DATA_DIR, layout.FUSION_SUBDIR)
    written = plots.plot_fusion_metric_grids(data, plot_metrics, out_dir)
    print(f"fusion diagnostics: {len(written)} plot(s) under {out_dir}")


try:
    _plot_fusion_diagnostics()
except Exception as exc:  # diagnostics are best-effort; never break fusion
    print(f"[diagnostics] fusion metric plots skipped: {exc!r}")
