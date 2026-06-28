"""Builders that materialize fake ``.pt`` pipeline artifacts on disk.

A ``config``-free copy of the ``_Builders`` helper that used to live only in
``tests/experiment/conftest.py``. Used by the per-stage tests to seed a stage's
inputs without running the upstream stages, and available to the golden harness
for the same purpose. Paths come from :mod:`common.layout`, the same single
source of truth the pipeline stages and ``common`` readers use, so the test
fixtures can never drift from the real layout.
"""
import os

import torch
from torch import nn

from common import layout
from common.mlp import AdvancedMLP
from common.testing.fakes import H, INTER, N_EXPERTS, FakeFusionGate


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

    def experts(self):
        for layer in self.layers:
            for i in range(self.n_experts):
                torch.save(expert_mlp(i),
                           _ensure_parent(layout.expert_path(self.data_dir, layer, i)))
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
