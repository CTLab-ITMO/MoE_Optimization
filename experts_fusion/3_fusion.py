"""Stage 3 -- fusion (§2.2, §3.4).

Replaces the old stage 6. For every group of every layer it merges the group's
experts with each registered fusion method, scores them on the group's held-out
validation calibration data, and writes one versioned
:class:`~common.fused_group.FusedExpert` per ``(method, layer, group)`` plus the
per-metric ``Best_<metric>`` winners.

Two pieces are lifted into ``common`` so stages 3 and 4 share one definition (the
metric compartment, §3.4): :func:`~common.mlp_utils.test_model_on_data` scores a
fused expert, and :func:`~common.mlp_utils.select_best_fused` picks the
``Best_<metric>`` winner with the ``next(iter(...))`` fallback (never a hardcoded
``"Pruning"``, §5).

Idempotency / resume (§3.4): each fused artifact records its upstream hash
(config hash + work-item coordinates + partition hash) in the provenance slot, so
a re-run skips a group whose every output is already fresh.
"""
import os

import torch

from common import layout, provenance
from common.artifacts import GroupActivationIndex, Partition
from common.context import StageContext
from common.fused_group import FusedExpert
from common.fusion import FUSION_REGISTRY
from common.metrics import METRIC_REGISTRY
from common.mlp_utils import (
    FUSION_METHODS,
    best_metrics,
    select_best_fused,
    test_model_on_data,
)
from common.names import BEST_PREFIX, TIME_METRIC
from common.plugins import load_plugins
from common.registry import require_all_registered
from common.seeding import set_seed
from common.serialization import make_provenance
from experts_fusion.config import (
    DATA_DIR,
    EXPERTS,
    FILTER_FUSIONS,
    FILTER_METRICS,
    LAYERS,
    LOAD_PLUGINS,
    MIN_PAIR_VALIDATION_SAMPLES,
    SEED,
    config_snapshot_dict,
    device,
)

# Pull in installed plugins (off by default), then validate the config filters
# against the registries and guard config drift from the stage-1 snapshot.
load_plugins(LOAD_PLUGINS)
require_all_registered(FILTER_FUSIONS, FUSION_REGISTRY, field="FILTER_FUSIONS")
require_all_registered(FILTER_METRICS, METRIC_REGISTRY, field="FILTER_METRICS")
provenance.assert_config_matches_snapshot(DATA_DIR, config_snapshot_dict())

fusion_methods = dict(FUSION_METHODS)
if FILTER_FUSIONS is not None:
    fusion_methods = {k: v for k, v in fusion_methods.items() if k in FILTER_FUSIONS}
selected_best_metrics = dict(best_metrics)
if FILTER_METRICS is not None:
    selected_best_metrics = {k: v for k, v in selected_best_metrics.items() if k in FILTER_METRICS}

set_seed(SEED)
# Base context: per-layer/per-group copies give the stochastic fusion (Fisher) an
# order-independent generator (S4, §3.3).
base_ctx = StageContext(seed=SEED, device=device)
config_fingerprint = provenance.config_hash(config_snapshot_dict())


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

# Per-scored-group reconstruction reports, collected for the best-effort
# diagnostic plots below (the descendant of the legacy stage-6 plotting block).
# ``resumed_any`` records whether any group was skipped as already-fresh: on a
# partial resume the plots would cover only the re-scored subset, so we skip
# plotting entirely rather than overwrite a complete plot with a partial one.
experiment_results = []
resumed_any = False

for layer in LAYERS:
    partition = Partition.load(layout.group_path(DATA_DIR, layer))
    partition_fingerprint = provenance.header_hash(layout.group_path(DATA_DIR, layer))
    train_index = GroupActivationIndex.load(
        layout.group_activations_path(DATA_DIR, layout.TRAIN_SPLIT, layer)
    ).to_lists()
    validation_index = GroupActivationIndex.load(
        layout.group_activations_path(DATA_DIR, layout.VALIDATION_SPLIT, layer)
    ).to_lists()
    experts = [
        torch.load(layout.expert_path(DATA_DIR, layer, i), weights_only=False)
        for i in range(EXPERTS)
    ]
    train_mlp_inputs = torch.load(layout.mlp_inputs_path(DATA_DIR, layout.TRAIN_SPLIT, layer))
    validation_mlp_inputs = torch.load(
        layout.mlp_inputs_path(DATA_DIR, layout.VALIDATION_SPLIT, layer)
    )

    for group_idx, group in enumerate(partition.groups):
        def upstream(name):
            return provenance.fused_upstream_hash(
                config_hash=config_fingerprint,
                layer=layer,
                method=name,
                group=group_idx,
                extra=partition_fingerprint,
            )

        # Resume: skip the whole group when every output is already fresh (the
        # Best_* selection needs all methods, so freshness is all-or-nothing).
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

        # k-agnostic public contract; for a pair this is the same merge as the
        # legacy ``method(a, b, train_data)`` positional form.
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
    """Best-effort distribution plots of the per-method reconstruction metrics.

    Descendant of the legacy stage-6 plotting block, but every output lands under
    ``DATA_DIR/plots`` (never CWD-relative ``plots/``; §1.4). Plotted only on a
    full (non-resumed) run, so a partial resume never overwrites a complete plot
    with a subset. Best-effort: the caller wraps this so any failure is printed
    and skipped rather than breaking the fusion stage.
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
