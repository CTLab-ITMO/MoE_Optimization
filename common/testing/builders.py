"""Builders that materialize fake ``.pt`` pipeline artifacts on disk.

A ``config``-free copy of the ``_Builders`` helper that used to live only in
``tests/experiment/conftest.py``. Used by the per-stage tests to seed a stage's
inputs without running the upstream stages, and available to the golden harness
for the same purpose. Layouts mirror the legacy pipeline exactly.
"""
import os

import torch
from torch import nn

from common.mlp import AdvancedMLP
from common.testing.fakes import H, INTER, N_EXPERTS, FakeFusionGate


def expert_mlp(seed=0):
    torch.manual_seed(seed)
    return AdvancedMLP(H, INTER, H, device=torch.device("cpu"))


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
            torch.save(m, self._path("activations", f"train_matrix_{layer}.pt"))
            torch.save(m.clone(), self._path("activations", f"validation_matrix_{layer}.pt"))

    def pairs(self):
        pairs = [(2 * i, 2 * i + 1) for i in range(self.n_experts // 2)]
        for layer in self.layers:
            torch.save(pairs, self._path("pairs", f"{layer}.pt"))
        return pairs

    def experts(self):
        for layer in self.layers:
            for i in range(self.n_experts):
                torch.save(expert_mlp(i), self._path("experts", str(layer), f"expert_{i}.pt"))
            torch.save(FakeFusionGate(), self._path("experts", str(layer), "gate.pt"))
            torch.save(nn.Linear(H, H, bias=False),
                       self._path("experts", str(layer), "shared_expert.pt"))
            torch.save(nn.Linear(H, 1, bias=False),
                       self._path("experts", str(layer), "shared_expert_gate.pt"))

    def mlp_inputs(self, n=16):
        for layer in self.layers:
            torch.save([torch.randn(H) for _ in range(n)],
                       self._path("mlp_inputs", f"train_{layer}.pt"))
            torch.save([torch.randn(H) for _ in range(n)],
                       self._path("mlp_inputs", f"validation_{layer}.pt"))

    def pairs_activations(self, pairs, n=16):
        for layer in self.layers:
            torch.save([list(range(n)) for _ in pairs],
                       self._path("activations", f"pairs_train_{layer}.pt"))
            torch.save([list(range(n)) for _ in pairs],
                       self._path("activations", f"pairs_validation_{layer}.pt"))

    def fusion_experts(self, name, pairs):
        for layer in self.layers:
            for i in range(len(pairs)):
                torch.save(expert_mlp(i),
                           self._path("fusion_experts", name, str(layer), f"{i}.pt"))
