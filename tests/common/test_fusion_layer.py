import os

import pytest
import torch
from torch import nn

from common.expert_adapter import NoSharedExpertAdapter, Qwen15SharedExpert
from common.fused_group import IdentityFusedGroup, MLPFusedGroup
from common.fusion_layer import (
    FusionLayer,
    get_fusion_layer_by_index,
)
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
    shared = Qwen15SharedExpert(
        nn.Linear(HIDDEN, HIDDEN, bias=False), nn.Linear(HIDDEN, 1, bias=False)
    )
    return FusionLayer(
        experts=experts,
        expert_groups=PAIRS,
        fusion_experts=fusion_experts,
        gate=gate,
        shared=shared,
        use_default_experts=use_default_experts,
    )


def test_group_maps():
    layer = _build_fusion_layer(use_default_experts=True)
    assert layer.num_experts == NUM_EXPERTS
    assert layer.group_idx[0] == 0 and layer.group_idx[3] == 1


def test_layer_wraps_fused_experts_in_fused_groups():
    """The reconstruction seam (§3.7): the layer drives ``FusedGroup.reconstruct``,
    so each fused expert is wrapped in a group referencing the same module."""
    layer = _build_fusion_layer(use_default_experts=False)
    assert len(layer.fusion_groups) == len(PAIRS)
    for group, expert in zip(layer.fusion_groups, layer.fusion_experts):
        assert isinstance(group, MLPFusedGroup)
        assert group.mlp is expert


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


# --------------------------------------------------------------------------- #
# Groups of size != 2 (identity reconstruction) and constructor validation.
# --------------------------------------------------------------------------- #
MIXED_GROUPS = [(0,), (1, 2, 3)]


def _build_mixed_layer():
    torch.manual_seed(0)
    gate = FakeGate()
    fusion_experts = [AdvancedMLP(HIDDEN, 6, HIDDEN, device=CPU) for _ in MIXED_GROUPS]
    fusion_groups = [
        IdentityFusedGroup(mlp, len(members))
        for mlp, members in zip(fusion_experts, MIXED_GROUPS)
    ]
    shared = Qwen15SharedExpert(
        nn.Linear(HIDDEN, HIDDEN, bias=False), nn.Linear(HIDDEN, 1, bias=False)
    )
    layer = FusionLayer(
        experts=[None] * NUM_EXPERTS,
        expert_groups=MIXED_GROUPS,
        fusion_experts=fusion_experts,
        gate=gate,
        shared=shared,
        use_default_experts=False,
        fusion_groups=fusion_groups,
    )
    return layer, gate, fusion_experts, shared


def test_group_maps_for_singleton_and_triple():
    layer, *_ = _build_mixed_layer()
    assert layer.group_idx == [0, 1, 1, 1]
    assert layer.expert_groups == [(0,), (1, 2, 3)]


def test_forward_singleton_and_triple_matches_manual_routing():
    """Every member of an identity group must be served by the group's fused
    output weighted by that member's own routing weight."""
    layer, gate, fusion_experts, shared = _build_mixed_layer()
    group_of = {0: 0, 1: 1, 2: 1, 3: 1}

    hidden_states = torch.randn(2, 3, HIDDEN)
    flat = hidden_states.view(-1, HIDDEN)
    with torch.no_grad():
        _, weights, selected = gate(flat)
        expected = shared(flat).clone()
        for t in range(flat.shape[0]):
            for j in range(TOP_K):
                expert = int(selected[t, j])
                fused = fusion_experts[group_of[expert]]
                expected[t] += weights[t, j] * fused(flat[t : t + 1])[0]
        out = layer.forward(hidden_states)
    assert torch.allclose(out, expected.view_as(hidden_states), atol=1e-6)


def test_constructor_rejects_group_size_mismatch():
    torch.manual_seed(0)
    fusion_experts = [AdvancedMLP(HIDDEN, 6, HIDDEN, device=CPU) for _ in MIXED_GROUPS]
    fusion_groups = [
        IdentityFusedGroup(fusion_experts[0], 1),
        IdentityFusedGroup(fusion_experts[1], 2),  # partition group has 3 members
    ]
    with pytest.raises(ValueError, match="group_size"):
        FusionLayer(
            experts=[None] * NUM_EXPERTS,
            expert_groups=MIXED_GROUPS,
            fusion_experts=fusion_experts,
            gate=FakeGate(),
            shared=Qwen15SharedExpert(
                nn.Linear(HIDDEN, HIDDEN, bias=False), nn.Linear(HIDDEN, 1, bias=False)
            ),
            use_default_experts=False,
            fusion_groups=fusion_groups,
        )


def test_constructor_without_fusion_groups_rejects_non_pairs():
    torch.manual_seed(0)
    fusion_experts = [AdvancedMLP(HIDDEN, 6, HIDDEN, device=CPU) for _ in MIXED_GROUPS]
    with pytest.raises(ValueError, match="must be a pair"):
        FusionLayer(
            experts=[None] * NUM_EXPERTS,
            expert_groups=MIXED_GROUPS,
            fusion_experts=fusion_experts,
            gate=FakeGate(),
            shared=Qwen15SharedExpert(
                nn.Linear(HIDDEN, HIDDEN, bias=False), nn.Linear(HIDDEN, 1, bias=False)
            ),
            use_default_experts=False,
        )


def test_constructor_rejects_groups_experts_length_mismatch():
    torch.manual_seed(0)
    with pytest.raises(ValueError, match="one entry per group"):
        FusionLayer(
            experts=[None] * NUM_EXPERTS,
            expert_groups=MIXED_GROUPS,
            fusion_experts=[AdvancedMLP(HIDDEN, 6, HIDDEN, device=CPU)],
            gate=FakeGate(),
            shared=Qwen15SharedExpert(
                nn.Linear(HIDDEN, HIDDEN, bias=False), nn.Linear(HIDDEN, 1, bias=False)
            ),
            use_default_experts=False,
        )


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
def data_dir(tmp_path):
    return str(tmp_path)


def test_get_fusion_layer_by_index_with_default_experts(data_dir):
    """``num_experts=None`` infers the expert count from the saved files."""
    _populate_data_dir(data_dir, index=0)
    layer = get_fusion_layer_by_index(0, "fusion_experts/ZipIt", data_dir, device=CPU,
                                      use_default_experts=True, num_experts=None)
    assert isinstance(layer, FusionLayer)
    assert layer.num_experts == NUM_EXPERTS
    out = layer.forward(torch.randn(1, 2, HIDDEN))
    assert out.shape == (1, 2, HIDDEN)


def test_get_fusion_layer_by_index_production_mode(data_dir):
    _populate_data_dir(data_dir, index=0)
    layer = get_fusion_layer_by_index(0, "fusion_experts/ZipIt", data_dir, device=CPU,
                                      use_default_experts=False, num_experts=NUM_EXPERTS)
    assert isinstance(layer, FusionLayer)
    assert not hasattr(layer, "experts")
    out = layer.forward(torch.randn(1, 2, HIDDEN))
    assert out.shape == (1, 2, HIDDEN)


def _shared_paths(data_dir, index=0):
    base = os.path.join(data_dir, "experts", str(index))
    return os.path.join(base, "shared_expert.pt"), os.path.join(base, "shared_expert_gate.pt")


def test_loader_shared_absent_falls_back_to_zeros(data_dir):
    """No shared files (a Mixtral-like family) -> the layer's shared term is zeros."""
    _populate_data_dir(data_dir, index=0)
    for path in _shared_paths(data_dir):
        os.remove(path)
    layer = get_fusion_layer_by_index(0, "fusion_experts/ZipIt", data_dir, device=CPU,
                                      use_default_experts=False, num_experts=NUM_EXPERTS)
    assert layer.shared.has_shared is False
    out = layer.forward(torch.randn(1, 2, HIDDEN))
    assert out.shape == (1, 2, HIDDEN)


def test_loader_shared_partial_is_corrupt(data_dir):
    """Exactly one shared file present is a corrupt tree -> explicit error, not silent zeros."""
    _populate_data_dir(data_dir, index=0)
    _, gate_path = _shared_paths(data_dir)
    os.remove(gate_path)
    with pytest.raises(FileNotFoundError, match="corrupt shared-expert"):
        get_fusion_layer_by_index(0, "fusion_experts/ZipIt", data_dir, device=CPU,
                                  use_default_experts=False, num_experts=NUM_EXPERTS)


def test_loader_threads_shared_adapter(data_dir):
    """The shared_expert_adapter override is honored: NoShared yields zeros even with files present."""
    _populate_data_dir(data_dir, index=0)
    layer = get_fusion_layer_by_index(0, "fusion_experts/ZipIt", data_dir, device=CPU,
                                      use_default_experts=False, num_experts=NUM_EXPERTS,
                                      shared_expert_adapter=NoSharedExpertAdapter())
    assert layer.shared.has_shared is False
