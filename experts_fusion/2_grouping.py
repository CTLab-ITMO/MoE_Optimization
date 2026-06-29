"""Stage 2 -- grouping + per-group activation join (§2.2, §3.4).

Collapses the old stage 2 (pairing) and the join half of the old stage 5 into a
pure-CPU stage that needs no model: changing the grouping method only re-runs
this stage, not the (expensive) capture in stage 1 -- the §3.4 cut line.

* **Grouping**: the registered :class:`~common.grouping.Grouping` strategy
  partitions each layer's experts into a :class:`~common.artifacts.Partition`,
  saved to the versioned ``groups/`` prefix (replacing the legacy pickled
  ``pairs/``). The default ``activation_correlation`` reads the stage-1
  co-activation matrix and reproduces the legacy pairing exactly (group order +
  tuple orientation).
* **Join**: for each split, the per-example top-k selections (stage 1) are joined
  with the partition into a :class:`~common.artifacts.GroupActivationIndex` --
  for each group, the (ascending) indices of the examples that activate it. This
  is the calibration index stage 3 gathers its per-group data from.
"""
import os

import torch

from common import layout, provenance
from common.artifacts import GroupActivationIndex
from common.context import StageContext
from common.grouping import GROUPING_REGISTRY
from common.plugins import load_plugins
from common.registry import require_registered
from common.seeding import set_seed
from experts_fusion.config import (
    DATA_DIR,
    EXPERTS,
    LAYERS,
    LOAD_PLUGINS,
    PAIRING_METHOD,
    SEED,
    config_snapshot_dict,
    device,
)

# Pull in installed plugins (off by default), then validate the grouping choice
# against the registry and guard against config drift from the stage-1 snapshot.
load_plugins(LOAD_PLUGINS)
require_registered(PAIRING_METHOD, GROUPING_REGISTRY, field="PAIRING_METHOD")
provenance.assert_config_matches_snapshot(DATA_DIR, config_snapshot_dict())

set_seed(SEED)

grouping = GROUPING_REGISTRY[PAIRING_METHOD]
# Base context: the run seed gives the stochastic strategies (random /
# output_similarity) an order-independent per-layer generator (S4); the default
# activation_correlation is deterministic and ignores it.
base_ctx = StageContext(seed=SEED, device=device, data_dir=DATA_DIR, n_experts=EXPERTS)


def join_group_activations(partition, topk_list) -> list[list[int]]:
    """For each group, the ascending example indices that activate it.

    Generalizes the old per-pair index build (stage 5) to k-agnostic groups:
    an expert -> group map turns each example's top-k selections into the set of
    groups it activated. Ascending order is deterministic; the deterministic
    fusion methods are order-invariant, so this matches the legacy join.
    """
    expert_to_group = [-1] * EXPERTS
    for group_idx, group in enumerate(partition.groups):
        for member in group.members:
            expert_to_group[member] = group_idx
    group_examples = [set() for _ in partition.groups]
    for example_idx, selection in enumerate(topk_list):
        for expert in selection:
            group_idx = expert_to_group[int(expert)]
            if group_idx >= 0:
                group_examples[group_idx].add(example_idx)
    return [sorted(examples) for examples in group_examples]


os.makedirs(layout.groups_dir(DATA_DIR), exist_ok=True)

for layer in LAYERS:
    partition = grouping.partition(layer, base_ctx.for_layer(layer))
    partition.save(layout.group_path(DATA_DIR, layer))
    for split in (layout.TRAIN_SPLIT, layout.VALIDATION_SPLIT):
        topk = torch.load(layout.activations_all_path(DATA_DIR, split, layer))
        index = GroupActivationIndex.from_lists(
            join_group_activations(partition, topk), split=split
        )
        index.save(layout.group_activations_path(DATA_DIR, split, layer))
    print(f"layer {layer}: {partition.num_groups} groups")
