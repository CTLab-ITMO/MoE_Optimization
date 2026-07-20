"""Tests for the REAP / HC-SMoE baseline grouping strategies.

Covers ``SaliencyPruneGrouping`` (REAP saliency pruning -> survivor singletons,
deterministic tie-break) and ``HCSMoEClusterGrouping`` (UPGMA over characteristic
activations -> clusters covering every expert), both registered on their
``PairingMethod`` names via ``common.baselines``.
"""

import numpy as np
import pytest
import torch

import common.baselines  # noqa: F401  ensure baseline strategies register
from common import layout
from common.baselines import signals
from common.baselines.hcsmoe import (
    HCSMoEClusterGrouping,
    _average_linkage,
    _linkage_to_labels,
)
from common.baselines.reap import SaliencyPruneGrouping
from common.context import StageContext
from common.grouping import GROUPING_REGISTRY
from common.mlp import AdvancedMLP
from common.names import PairingMethod

CPU = torch.device("cpu")
SEED = 1234

REAP = str(PairingMethod.REAP_SALIENCY)
HCSMOE = str(PairingMethod.HCSMOE_CLUSTERING)


@pytest.fixture
def allow_pickled_load(monkeypatch):
    """Let ``torch.load`` read the pickled fake experts the helpers save."""
    original = torch.load

    def loader(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "load", loader)


def _save_experts(data_dir, layer, n_experts, hidden=4, inter=6):
    import os

    os.makedirs(layout.experts_dir(data_dir, layer), exist_ok=True)
    for i in range(n_experts):
        torch.manual_seed(i)
        expert = AdvancedMLP(hidden, inter, hidden, device=CPU)
        torch.save(expert, layout.expert_path(data_dir, layer, i))


def _save_inputs(data_dir, layer, inputs):
    import os

    os.makedirs(layout.mlp_inputs_dir(data_dir), exist_ok=True)
    torch.save(inputs, layout.mlp_inputs_path(data_dir, layout.TRAIN_SPLIT, layer))


def _save_gate(data_dir, layer, hidden, n_experts, seed=7):
    import os

    from torch import nn

    torch.manual_seed(seed)
    os.makedirs(layout.experts_dir(data_dir, layer), exist_ok=True)
    torch.save(nn.Linear(hidden, n_experts, bias=False), layout.gate_path(data_dir, layer))


def _save_topk(data_dir, layer, selections):
    import os

    os.makedirs(layout.activations_dir(data_dir), exist_ok=True)
    torch.save(selections, layout.activations_all_path(data_dir, layout.TRAIN_SPLIT, layer))


def _ctx(data_dir=None, n_experts=None, layer=0, seed=SEED):
    return StageContext(
        seed=seed, device=CPU, data_dir=data_dir, n_experts=n_experts
    ).for_layer(layer)


# --------------------------------------------------------------------------- #
# SaliencyPruneGrouping (REAP): survivors as singletons, deterministic tie-break.
# --------------------------------------------------------------------------- #
def _patch_saliency(monkeypatch, saliency):
    """Force compute_reap_saliency to return a fixed per-expert score vector,
    so the grouping's partition/tie-break logic is tested in isolation."""

    def fake(layer, data_dir, n_experts, device, num_samples, batch_size,
             generator=None, split=layout.TRAIN_SPLIT):
        return torch.as_tensor(saliency, dtype=torch.float32)

    monkeypatch.setattr(signals, "compute_reap_saliency", fake)


def test_saliency_grouping_keeps_top_ratio_as_singletons(monkeypatch):
    _patch_saliency(monkeypatch, [0.5, 0.1, 0.9, 0.2])
    partition = SaliencyPruneGrouping(REAP, ratio=0.5).partition(0, _ctx("unused", 4))
    # int(4 * 0.5) = 2 dropped: idx 1 (0.1), idx 3 (0.2). Survivors 0, 2.
    assert [group.members for group in partition.groups] == [(0,), (2,)]
    assert all(group.size == 1 for group in partition.groups)
    assert partition.method == REAP


def test_saliency_grouping_registered_under_reap_name(monkeypatch):
    _patch_saliency(monkeypatch, [0.9, 0.1, 0.5, 0.2])
    partition = GROUPING_REGISTRY[REAP].partition(0, _ctx("unused", 4))
    assert partition.method == REAP
    assert all(group.size == 1 for group in partition.groups)
    # Default ratio 0.5 -> drop idx 1 (0.1), idx 3 (0.2); survivors 0, 2.
    assert [group.members[0] for group in partition.groups] == [0, 2]


def test_saliency_grouping_tie_break_drops_lower_index_first(monkeypatch):
    # Three experts tie at 0.5; n_pruned=2 drops the two lowest indices (1, 2)
    # and keeps index 3 -- proving the stable-argsort rule, not torch.topk order.
    _patch_saliency(monkeypatch, [1.0, 0.5, 0.5, 0.5])
    partition = SaliencyPruneGrouping(REAP, ratio=0.5).partition(0, _ctx("unused", 4))
    assert [group.members[0] for group in partition.groups] == [0, 3]


def test_saliency_grouping_never_routed_dropped_first(monkeypatch):
    # A never-routed expert scores 0 (lowest possible) and is dropped first, even
    # though a routed expert (idx 2, 0.3) scores less than others.
    _patch_saliency(monkeypatch, [0.0, 0.7, 0.3, 0.9])
    partition = SaliencyPruneGrouping(REAP, ratio=0.25).partition(0, _ctx("unused", 4))
    # int(4 * 0.25) = 1 dropped -> only the zero (idx 0). Survivors 1, 2, 3.
    assert [group.members[0] for group in partition.groups] == [1, 2, 3]


def test_saliency_grouping_dropped_experts_absent_from_partition(monkeypatch):
    _patch_saliency(monkeypatch, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    partition = SaliencyPruneGrouping(REAP, ratio=0.5).partition(0, _ctx("unused", 6))
    covered = {member for group in partition.groups for member in group.members}
    # int(6 * 0.5) = 3 dropped (idx 0, 1, 2); survivors 3, 4, 5 are the only covered.
    assert covered == {3, 4, 5}
    assert {0, 1, 2}.isdisjoint(covered)


def test_saliency_grouping_requires_n_experts(monkeypatch):
    _patch_saliency(monkeypatch, [1.0, 2.0])
    with pytest.raises(ValueError, match="n_experts"):
        SaliencyPruneGrouping(REAP).partition(0, _ctx("unused", n_experts=None))


def test_saliency_grouping_end_to_end_produces_singletons(tmp_path, allow_pickled_load):
    # Full wiring: real stage-1 artifacts -> compute_reap_saliency -> singletons.
    # Experts 2 and 3 are never routed (saliency 0) so ratio 0.5 drops exactly them.
    data_dir = str(tmp_path)
    n, hidden = 4, 4
    _save_experts(data_dir, 0, n, hidden=hidden)
    _save_gate(data_dir, 0, hidden, n)
    _save_inputs(data_dir, 0, torch.randn(12, hidden))
    _save_topk(data_dir, 0, [torch.tensor([0, 1]) for _ in range(12)])

    partition = GROUPING_REGISTRY[REAP].partition(0, _ctx(data_dir, n))

    assert partition.method == REAP
    assert all(group.size == 1 for group in partition.groups)
    covered = {member for group in partition.groups for member in group.members}
    assert covered == {0, 1}


# --------------------------------------------------------------------------- #
# HCSMoEClusterGrouping (HC-SMoE): UPGMA over characteristic activations.
# --------------------------------------------------------------------------- #
# Four points on a line at 0, 1, 2, 10 -> pairwise |x - y| distance matrix:
#     [[ 0, 1, 2, 10],
#      [ 1, 0, 1,  9],
#      [ 2, 1, 0,  8],
#      [10, 9, 8,  0]]
_LINE_FEATURES = [[0.0], [1.0], [2.0], [10.0]]
_LINE_DISTANCES = torch.tensor(
    [[0.0, 1.0, 2.0, 10.0],
     [1.0, 0.0, 1.0, 9.0],
     [2.0, 1.0, 0.0, 8.0],
     [10.0, 9.0, 8.0, 0.0]]
)


def _patch_characteristic(monkeypatch, features):
    """Force compute_characteristic_activation to return fixed per-expert feature
    rows, isolating the clustering/cut logic from the dense-forward compute."""

    def fake(layer, data_dir, n_experts, device, num_samples, batch_size,
             generator=None, split=layout.TRAIN_SPLIT):
        return torch.as_tensor(features, dtype=torch.float32)

    monkeypatch.setattr(signals, "compute_characteristic_activation", fake)


def test_average_linkage_is_upgma_size_weighted_closed_form():
    # Hand-computed UPGMA on _LINE_DISTANCES:
    #   merge (0,1) @ d=1   -> cluster 4, size 2
    #   merge (2,4) @ d=1.5 -> cluster 5, size 3   [d(4,2)=(2+1)/2=1.5]
    #   merge (3,5) @ d=9   -> cluster 6, size 4
    # The last distance is the UPGMA/WPGMA discriminator: size-weighted
    #   d(5,3)=(1*8 + 2*9.5)/3 = 9.0, whereas naive WPGMA (d(2,3)+d(4,3))/2 = 8.75.
    linkage = _average_linkage(_LINE_DISTANCES)
    expected = np.array([[0.0, 1.0, 1.0, 2.0],
                         [2.0, 4.0, 1.5, 3.0],
                         [3.0, 5.0, 9.0, 4.0]])
    assert np.allclose(linkage, expected)
    # Rows are ordered idx_i < idx_j and new clusters index from n.
    assert (linkage[:, 0] < linkage[:, 1]).all()


def test_linkage_to_labels_cut_yields_triple_and_singleton():
    linkage = _average_linkage(_LINE_DISTANCES)
    labels = _linkage_to_labels(linkage, num_clusters=2)
    # Experts 0,1,2 land in one cluster, expert 3 alone.
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] != labels[0]
    assert set(labels.tolist()) == {0, 1}


def test_hcsmoe_grouping_produces_singleton_and_cluster(monkeypatch):
    _patch_characteristic(monkeypatch, _LINE_FEATURES)
    partition = HCSMoEClusterGrouping(HCSMOE, ratio=0.5).partition(0, _ctx("unused", 4))
    # int(4 * (1 - 0.5)) = 2 clusters: the {0,1,2} triple and the {3} singleton.
    assert [group.members for group in partition.groups] == [(0, 1, 2), (3,)]
    assert partition.method == HCSMOE
    sizes = sorted(group.size for group in partition.groups)
    assert sizes == [1, 3]


def test_hcsmoe_grouping_covers_every_expert(monkeypatch):
    _patch_characteristic(monkeypatch, _LINE_FEATURES)
    partition = HCSMoEClusterGrouping(HCSMOE, ratio=0.5).partition(0, _ctx("unused", 4))
    covered = [member for group in partition.groups for member in group.members]
    # Unlike REAP pruning, HC-SMoE leaves no expert uncovered (router stays intact).
    assert sorted(covered) == [0, 1, 2, 3]


def test_hcsmoe_grouping_is_deterministic(monkeypatch):
    _patch_characteristic(monkeypatch, _LINE_FEATURES)
    strategy = HCSMoEClusterGrouping(HCSMOE, ratio=0.5)
    first = strategy.partition(0, _ctx("unused", 4))
    second = strategy.partition(0, _ctx("unused", 4))
    assert [g.members for g in first.groups] == [g.members for g in second.groups]


def test_hcsmoe_grouping_registered_under_hcsmoe_name(monkeypatch):
    _patch_characteristic(monkeypatch, _LINE_FEATURES)
    partition = GROUPING_REGISTRY[HCSMOE].partition(0, _ctx("unused", 4))
    assert partition.method == HCSMOE
    # Default ratio 0.5 keeps int(4 * 0.5) = 2 clusters.
    assert partition.num_groups == 2


def test_hcsmoe_grouping_requires_n_experts(monkeypatch):
    _patch_characteristic(monkeypatch, _LINE_FEATURES)
    with pytest.raises(ValueError, match="n_experts"):
        HCSMoEClusterGrouping(HCSMOE).partition(0, _ctx("unused", n_experts=None))


def test_hcsmoe_grouping_end_to_end_covers_all_experts(tmp_path, allow_pickled_load):
    # Full wiring: real stage-1 artifacts -> compute_characteristic_activation ->
    # euclidean distances -> UPGMA -> clusters covering every expert.
    data_dir = str(tmp_path)
    n, hidden = 6, 4
    _save_experts(data_dir, 0, n, hidden=hidden)
    _save_inputs(data_dir, 0, torch.randn(16, hidden))

    partition = GROUPING_REGISTRY[HCSMOE].partition(0, _ctx(data_dir, n))

    assert partition.method == HCSMOE
    assert partition.num_groups == int(n * 0.5)  # 3 clusters survive
    covered = sorted(member for group in partition.groups for member in group.members)
    assert covered == list(range(n))
