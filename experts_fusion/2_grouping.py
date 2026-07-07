"""Stage 2: group experts then build per-group activation indices. CPU-only.

Changing the grouping method only requires re-running this stage (no model needed).

Grouping: the configured strategy (default: activation_correlation) partitions
each layer's experts into a Partition saved under groups/. Order is deterministic.
Join: top-k expert selections are mapped to groups, producing a GroupActivationIndex
(sorted example indices per group) that stage 3 uses for calibration.
"""
import os

import torch

from common import layout, provenance
from common.artifacts import GroupActivationIndex
from common.context import StageContext
from common.grouping import GROUPING_REGISTRY
from common.registry import require_registered
from common.seeding import set_seed
from experts_fusion.config import (
    DATA_DIR,
    EXPERTS,
    LAYERS,
    PAIRING_METHOD,
    SEED,
    config_snapshot_dict,
    device,
)

# Validate PAIRING_METHOD is registered and config matches the stage-1 snapshot.
require_registered(PAIRING_METHOD, GROUPING_REGISTRY, field="PAIRING_METHOD")
provenance.assert_config_matches_snapshot(DATA_DIR, config_snapshot_dict())
layout.validate_layout_version(DATA_DIR)

set_seed(SEED)

grouping = GROUPING_REGISTRY[PAIRING_METHOD]
# Seed drives stochastic strategies (random/output_similarity); the default
# activation_correlation is deterministic and ignores it.
base_ctx = StageContext(seed=SEED, device=device, data_dir=DATA_DIR, n_experts=EXPERTS)


def join_group_activations(partition, topk_list) -> list[list[int]]:
    """Return sorted example indices that activate each group.

    Builds an expert->group map, then walks each example's top-k selections
    to collect activating examples per group. Sorted order is deterministic.
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

# Iterate the MoE layers stage 1 actually produced (from the run manifest), not
# the raw config LAYERS: a dense layer in LAYERS is skipped by stage 1 and has no
# artifacts here (CR#4). Falls back to LAYERS when no manifest is present.
produced_layers = provenance.resolve_produced_layers(DATA_DIR, LAYERS)

for layer in produced_layers:
    partition = grouping.partition(layer, base_ctx.for_layer(layer))
    partition.save(layout.group_path(DATA_DIR, layer))
    for split in (layout.TRAIN_SPLIT, layout.VALIDATION_SPLIT):
        topk = torch.load(layout.activations_all_path(DATA_DIR, split, layer))
        index = GroupActivationIndex.from_lists(
            join_group_activations(partition, topk), split=split
        )
        index.save(layout.group_activations_path(DATA_DIR, split, layer))
    print(f"layer {layer}: {partition.num_groups} groups")
