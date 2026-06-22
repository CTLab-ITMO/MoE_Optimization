import os

import pytest
import torch
from torch import nn

from common import fusion_layer
from common.fusion_layer import FusionLayer, get_fusion_layer_by_index
from common.mlp import AdvancedMLP

CPU = torch.device("cpu")
HIDDEN = 4
NUM_EXPERTS = 4
TOP_K = 2
PAIRS = [(0, 1), (2, 3)]


class FakeGate(nn.Module):
    def __init__(self, num_experts=NUM_EXPERTS, top_k=TOP_K, hidden=HIDDEN):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.proj = nn.Linear(hidden, num_experts, bias=False)

    def forward(self, x):
        logits = self.proj(x)
        weights, selected = torch.topk(logits.softmax(dim=-1), self.top_k, dim=-1)
        return logits, weights, selected


def _build_fusion_layer(use_default_experts):
    torch.manual_seed(0)
    gate = FakeGate()
    experts = [nn.Linear(HIDDEN, HIDDEN, bias=False) for _ in range(NUM_EXPERTS)]
    fusion_experts = [AdvancedMLP(HIDDEN, 6, HIDDEN, device=CPU) for _ in PAIRS]
    shared_expert = nn.Linear(HIDDEN, HIDDEN, bias=False)
    shared_expert_gate = nn.Linear(HIDDEN, 1, bias=False)
    return FusionLayer(
        experts=experts,
        expert_pairs=PAIRS,
        fusion_experts=fusion_experts,
        gate=gate,
        shared_expert=shared_expert,
        shared_expert_gate=shared_expert_gate,
        use_default_experts=use_default_experts,
    )


def test_pair_maps():
    layer = _build_fusion_layer(use_default_experts=True)
    assert layer.num_experts == NUM_EXPERTS
    assert layer.pair_idx[0] == 0 and layer.pair_idx[3] == 1


def test_forward_production_keeps_shape():
    layer = _build_fusion_layer(use_default_experts=False)
    hidden_states = torch.randn(2, 3, HIDDEN)
    out = layer.forward(hidden_states)
    assert out.shape == hidden_states.shape
    assert torch.isfinite(out).all()


def test_forward_default_keeps_shape():
    layer = _build_fusion_layer(use_default_experts=True)
    hidden_states = torch.randn(2, 3, HIDDEN)
    out = layer.forward_default(hidden_states)
    assert out.shape == hidden_states.shape
    assert torch.isfinite(out).all()


def _populate_data_dir(root, index, experts_subdir="fusion_experts/ZipIt"):
    torch.manual_seed(0)
    pairs_dir = os.path.join(root, "pairs")
    experts_dir = os.path.join(root, "experts", str(index))
    fusion_dir = os.path.join(root, experts_subdir, str(index))
    for d in (pairs_dir, experts_dir, fusion_dir):
        os.makedirs(d, exist_ok=True)

    torch.save(PAIRS, os.path.join(pairs_dir, f"{index}.pt"))
    for i in range(NUM_EXPERTS):
        torch.save(nn.Linear(HIDDEN, HIDDEN, bias=False),
                   os.path.join(experts_dir, f"expert_{i}.pt"))
    for i in range(len(PAIRS)):
        torch.save(AdvancedMLP(HIDDEN, 6, HIDDEN, device=CPU),
                   os.path.join(fusion_dir, f"{i}.pt"))
    torch.save(FakeGate(), os.path.join(experts_dir, "gate.pt"))
    torch.save(nn.Linear(HIDDEN, HIDDEN, bias=False),
               os.path.join(experts_dir, "shared_expert.pt"))
    torch.save(nn.Linear(HIDDEN, 1, bias=False),
               os.path.join(experts_dir, "shared_expert_gate.pt"))


@pytest.fixture
def patched_config(monkeypatch, tmp_path):
    monkeypatch.setattr(fusion_layer, "EXPERTS", NUM_EXPERTS)
    return str(tmp_path)


def test_get_fusion_layer_by_index_with_default_experts(patched_config):
    _populate_data_dir(patched_config, index=0)
    layer = get_fusion_layer_by_index(0, "fusion_experts/ZipIt", patched_config, device=CPU,
                                      use_default_experts=True)
    assert isinstance(layer, FusionLayer)
    out = layer.forward(torch.randn(1, 2, HIDDEN))
    assert out.shape == (1, 2, HIDDEN)


def test_get_fusion_layer_by_index_production_mode(patched_config):
    _populate_data_dir(patched_config, index=0)
    layer = get_fusion_layer_by_index(0, "fusion_experts/ZipIt", patched_config, device=CPU,
                                      use_default_experts=False)
    assert isinstance(layer, FusionLayer)
    assert not hasattr(layer, "experts")
    out = layer.forward(torch.randn(1, 2, HIDDEN))
    assert out.shape == (1, 2, HIDDEN)
