"""Tests for the fused-group runtime contract and its serialization (S3).

Covers the S3 silent-break gate:

* ``MLPFusedGroup.reconstruct(x, member)`` is *exactly* ``forward_unzip(x,
  out=member)`` -- the contract is one implementation of the legacy path;
* member bounds / ``group_size`` of the pairwise group;
* ``FusedExpert`` round-trips through the S2 serializer forward-identically and
  carries the reconstruction-contract metadata;
* ``FusionLayer`` output is bit-identical whether its pairs come from a raw
  ``[(i, j)]`` list or a ``Partition`` -- and *differs* when the orientation is
  flipped (proving the equality test actually guards orientation).
"""

import pytest
import torch
from torch import nn

from common.artifacts import Partition
from common.fused_group import (
    FusedExpert,
    FusedGroup,
    MLPFusedGroup,
    ZIP_UNMERGE_RECONSTRUCTION,
)
from common.fusion import build_fusion_methods
from common.fusion_layer import FusionLayer
from common.mlp import AdvancedMLP
from common.serialization import SerializationError

CPU = torch.device("cpu")
HIDDEN = 4
INTER = 6
NUM_EXPERTS = 4
TOP_K = 2
PAIRS = [(0, 1), (2, 3)]


def _fused_with_u(seed_a=1, seed_b=2, method="ZipIt"):
    """A fused AdvancedMLP that has gone through the zip pipeline (``U`` set)."""
    torch.manual_seed(seed_a)
    a = AdvancedMLP(HIDDEN, INTER, HIDDEN, device=CPU)
    torch.manual_seed(seed_b)
    b = AdvancedMLP(HIDDEN, INTER, HIDDEN, device=CPU)
    return build_fusion_methods()[method](a, b, torch.randn(40, HIDDEN))


# --------------------------------------------------------------------------- #
# Abstract contract.
# --------------------------------------------------------------------------- #
def test_fused_group_is_abstract():
    with pytest.raises(TypeError):
        FusedGroup()


# --------------------------------------------------------------------------- #
# reconstruct == forward_unzip (the contract is one implementation).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", ["ZipIt", "Average", "Pruning", "LS_ZipIt"])
def test_reconstruct_is_exactly_forward_unzip(method):
    fused = _fused_with_u(method=method)
    group = MLPFusedGroup(fused)
    x = torch.randn(7, HIDDEN)
    with torch.no_grad():
        for member in range(group.group_size):
            assert torch.equal(
                group.reconstruct(x, member), fused.forward_unzip(x, out=member)
            )


def test_group_size_is_two():
    group = MLPFusedGroup(_fused_with_u())
    assert group.group_size == 2


def test_reconstruct_rejects_out_of_range_member():
    group = MLPFusedGroup(_fused_with_u())
    x = torch.randn(3, HIDDEN)
    with pytest.raises(IndexError):
        group.reconstruct(x, 2)
    with pytest.raises(IndexError):
        group.reconstruct(x, -1)


def test_reconstruct_rejects_non_int_member():
    group = MLPFusedGroup(_fused_with_u())
    x = torch.randn(3, HIDDEN)
    with pytest.raises(TypeError):
        group.reconstruct(x, 0.0)
    with pytest.raises(TypeError):
        group.reconstruct(x, True)  # bool is not a valid member


def test_mlp_fused_group_rejects_non_mlp():
    with pytest.raises(TypeError):
        MLPFusedGroup(nn.Linear(HIDDEN, HIDDEN))


# --------------------------------------------------------------------------- #
# FusedExpert serialization (S2 backbone, S3 contract metadata).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", ["ZipIt", "Average", "Pruning", "LS_ZipIt"])
def test_fused_expert_roundtrips_forward_identical(tmp_path, method):
    fused = _fused_with_u(method=method)
    expert = FusedExpert.from_group(MLPFusedGroup(fused))
    path = tmp_path / "fused.safetensors"
    expert.save(path)

    loaded = FusedExpert.load(path, device=CPU)
    assert loaded.group_size == 2
    assert loaded.reconstruction == ZIP_UNMERGE_RECONSTRUCTION

    group = loaded.to_group()
    x = torch.randn(7, HIDDEN)
    with torch.no_grad():
        for member in range(group.group_size):
            assert torch.allclose(
                group.reconstruct(x, member), fused.forward_unzip(x, out=member), atol=1e-6
            )


def test_fused_expert_artifact_header_carries_contract():
    expert = FusedExpert.from_group(MLPFusedGroup(_fused_with_u()))
    header = expert.to_artifact().header
    assert header["group_size"] == 2
    assert header["reconstruction"] == ZIP_UNMERGE_RECONSTRUCTION
    # The AdvancedMLP shape/activation header is still present.
    assert header["activation"] == "SiLU"
    assert header["in_features"] == HIDDEN


def test_fused_expert_rejects_unknown_reconstruction():
    with pytest.raises(SerializationError):
        FusedExpert(_fused_with_u(), reconstruction="not_a_real_kind")


def test_fused_expert_rejects_non_pairwise_group_size():
    with pytest.raises(SerializationError):
        FusedExpert(_fused_with_u(), group_size=3)


def test_from_artifact_rejects_unknown_reconstruction():
    expert = FusedExpert.from_group(MLPFusedGroup(_fused_with_u()))
    artifact = expert.to_artifact()
    artifact.header["reconstruction"] = "future_kind"
    with pytest.raises(SerializationError):
        FusedExpert.from_artifact(artifact, device=CPU)


# --------------------------------------------------------------------------- #
# FusionLayer output equality: raw pairs vs Partition, and the orientation guard.
# --------------------------------------------------------------------------- #
class _FakeGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_experts = NUM_EXPERTS
        self.proj = nn.Linear(HIDDEN, NUM_EXPERTS, bias=False)

    def forward(self, x):
        logits = self.proj(x)
        weights, selected = torch.topk(logits.softmax(dim=-1), TOP_K, dim=-1)
        return logits, weights, selected


def _build_layer(expert_pairs, fusion_experts, gate, shared_expert, shared_expert_gate):
    return FusionLayer(
        experts=[None] * NUM_EXPERTS,
        expert_pairs=expert_pairs,
        fusion_experts=fusion_experts,
        gate=gate,
        shared_expert=shared_expert,
        shared_expert_gate=shared_expert_gate,
        use_default_experts=False,
    )


def _shared_layer_parts():
    torch.manual_seed(0)
    fusion_experts = [_fused_with_u(seed_a=10 + i, seed_b=20 + i) for i in range(len(PAIRS))]
    gate = _FakeGate()
    shared_expert = nn.Linear(HIDDEN, HIDDEN, bias=False)
    shared_expert_gate = nn.Linear(HIDDEN, 1, bias=False)
    return fusion_experts, gate, shared_expert, shared_expert_gate


def test_fusion_layer_output_identical_pairs_vs_partition():
    parts = _shared_layer_parts()
    layer_pairs = _build_layer(PAIRS, *parts)
    layer_partition = _build_layer(Partition.from_pairs(PAIRS).to_pairs(), *parts)

    hidden = torch.randn(2, 3, HIDDEN)
    with torch.no_grad():
        out_pairs = layer_pairs.forward(hidden)
        out_partition = layer_partition.forward(hidden)
    assert torch.equal(out_pairs, out_partition)


def test_flipped_orientation_changes_fusion_layer_output():
    """Adversarial: flipping member order must change the output, so the equality
    test above is actually guarding orientation, not trivially passing."""
    parts = _shared_layer_parts()
    flipped = [(b, a) for (a, b) in PAIRS]
    layer_pairs = _build_layer(PAIRS, *parts)
    layer_flipped = _build_layer(flipped, *parts)

    hidden = torch.randn(2, 3, HIDDEN)
    with torch.no_grad():
        out_pairs = layer_pairs.forward(hidden)
        out_flipped = layer_flipped.forward(hidden)
    assert not torch.equal(out_pairs, out_flipped)
