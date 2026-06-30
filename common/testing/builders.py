"""Fake artifact builders for pipeline tests.

Writes ``.pt`` files (experts, matrices, partitions, fused groups, etc.) using
``common.layout`` paths, so tests can seed a stage's inputs without running the
upstream stages. Sharing the real layout paths prevents fixture drift.
"""
import os

import torch
from torch import nn

from common import layout
from common.artifacts import CoActivationMatrix, GroupActivationIndex, Partition, SavedExpert
from common.fused_group import FusedExpert
from common.mlp import AdvancedMLP
from common.testing.fakes import H, INTER, N_EXPERTS, TOP_K, FakeFusionGate


def expert_mlp(seed=0):
    torch.manual_seed(seed)
    return AdvancedMLP(H, INTER, H, device=torch.device("cpu"))


def _ensure_parent(path):
    """Create the parent directory of ``path`` and return ``path`` unchanged."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


class Builders:
    """Generate fake ``.pt`` artifacts."""

    def __init__(self, data_dir, layers, n_experts):
        self.data_dir = data_dir
        self.layers = layers
        self.n_experts = n_experts

    def _path(self, *parts):
        path = os.path.join(self.data_dir, *parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def ensure_dir(self, *parts):
        os.makedirs(os.path.join(self.data_dir, *parts), exist_ok=True)

    def activation_matrices(self):
        for layer in self.layers:
            m = torch.rand(self.n_experts, self.n_experts)
            m = m + m.t()
            torch.save(m, _ensure_parent(
                layout.activation_matrix_path(self.data_dir, layout.TRAIN_SPLIT, layer)))
            torch.save(m.clone(), _ensure_parent(
                layout.activation_matrix_path(self.data_dir, layout.VALIDATION_SPLIT, layer)))

    def pairs(self):
        pairs = [(2 * i, 2 * i + 1) for i in range(self.n_experts // 2)]
        for layer in self.layers:
            torch.save(pairs, _ensure_parent(layout.pairs_path(self.data_dir, layer)))
        return pairs

    def experts(self, saved=False):
        # saved=True: versioned SavedExpert; False (default): legacy pickle.
        # Gate and shared expert are always pickled.
        for layer in self.layers:
            for i in range(self.n_experts):
                path = _ensure_parent(layout.expert_path(self.data_dir, layer, i))
                if saved:
                    SavedExpert(expert_mlp(i)).save(path)
                else:
                    torch.save(expert_mlp(i), path)
            torch.save(FakeFusionGate(),
                       _ensure_parent(layout.gate_path(self.data_dir, layer)))
            torch.save(nn.Linear(H, H, bias=False),
                       _ensure_parent(layout.shared_expert_path(self.data_dir, layer)))
            torch.save(nn.Linear(H, 1, bias=False),
                       _ensure_parent(layout.shared_expert_gate_path(self.data_dir, layer)))

    def mlp_inputs(self, n=16):
        for layer in self.layers:
            torch.save([torch.randn(H) for _ in range(n)],
                       _ensure_parent(layout.mlp_inputs_path(self.data_dir, layout.TRAIN_SPLIT, layer)))
            torch.save([torch.randn(H) for _ in range(n)],
                       _ensure_parent(layout.mlp_inputs_path(self.data_dir, layout.VALIDATION_SPLIT, layer)))

    def pairs_activations(self, pairs, n=16):
        for layer in self.layers:
            torch.save([list(range(n)) for _ in pairs],
                       _ensure_parent(layout.pair_activations_path(self.data_dir, layout.TRAIN_SPLIT, layer)))
            torch.save([list(range(n)) for _ in pairs],
                       _ensure_parent(layout.pair_activations_path(self.data_dir, layout.VALIDATION_SPLIT, layer)))

    def fusion_experts(self, name, pairs):
        for layer in self.layers:
            for i in range(len(pairs)):
                torch.save(expert_mlp(i),
                           _ensure_parent(layout.fusion_expert_path(self.data_dir, name, layer, i)))

    # ----------------------------------------------------------------------- #
    # Versioned artifact builders (new module format).
    # ----------------------------------------------------------------------- #
    def coactivation_matrices(self):
        """Write fake ``CoActivationMatrix`` artifacts for train and validation splits."""
        for layer in self.layers:
            m = torch.rand(self.n_experts, self.n_experts)
            m = m + m.t()
            CoActivationMatrix(m, split=layout.TRAIN_SPLIT, sample_size=8).save(
                _ensure_parent(layout.activation_matrix_path(self.data_dir, layout.TRAIN_SPLIT, layer)))
            CoActivationMatrix(m.clone(), split=layout.VALIDATION_SPLIT, sample_size=8).save(
                _ensure_parent(layout.activation_matrix_path(self.data_dir, layout.VALIDATION_SPLIT, layer)))

    def topk_all(self, n=16):
        """Write per-example top-k expert selection tensors for both splits."""
        for layer in self.layers:
            for split in (layout.TRAIN_SPLIT, layout.VALIDATION_SPLIT):
                selections = [
                    torch.tensor([(i + k) % self.n_experts for k in range(TOP_K)])
                    for i in range(n)
                ]
                torch.save(selections, _ensure_parent(
                    layout.activations_all_path(self.data_dir, split, layer)))

    def partition(self, pairs=None):
        """Write a ``Partition`` artifact to ``groups/`` for each layer."""
        if pairs is None:
            pairs = [(2 * i, 2 * i + 1) for i in range(self.n_experts // 2)]
        for layer in self.layers:
            Partition.from_pairs(pairs, method="activation_correlation").save(
                _ensure_parent(layout.group_path(self.data_dir, layer)))
        return pairs

    def group_activations(self, n_groups, n=16):
        """Write ``GroupActivationIndex`` artifacts for both splits."""
        for layer in self.layers:
            for split in (layout.TRAIN_SPLIT, layout.VALIDATION_SPLIT):
                GroupActivationIndex.from_lists(
                    [list(range(n)) for _ in range(n_groups)], split=split
                ).save(_ensure_parent(
                    layout.group_activations_path(self.data_dir, split, layer)))

    def fused_experts_v1(self, name, pairs):
        """Write ``FusedExpert`` artifacts for each group in each layer."""
        for layer in self.layers:
            for i in range(len(pairs)):
                FusedExpert(expert_mlp(i)).save(
                    _ensure_parent(layout.fusion_expert_path(self.data_dir, name, layer, i)))
