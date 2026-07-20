"""Tests for the meaningful stored-data artifact types (S3, §2.4).

Covers:

* ``Partition.from_pairs`` is order- and orientation-preserving
  (``groups[i].members == (pairs[i][0], pairs[i][1])``) -- the silent-break
  invariant -- and ``to_pairs`` round-trips it;
* every data artifact (``Partition`` / ``GroupActivationIndex`` /
  ``CoActivationMatrix``) round-trips through the S2 serializer;
* the legacy ``pairs/N.pt`` pickle migrates into a ``Partition`` with orientation
  and order intact (S2 migration tied to the S3 ``Partition``).
"""

import pytest
import torch

from common import serialization as S
from common.artifacts import (
    CoActivationMatrix,
    Group,
    GroupActivationIndex,
    Partition,
    SavedExpert,
)
from common.mlp import AdvancedMLP

CPU = torch.device("cpu")


def test_saved_expert_round_trips_forward_identically(tmp_path):
    torch.manual_seed(0)
    mlp = AdvancedMLP(4, 6, 4, device=CPU)
    path = tmp_path / "expert_0.pt"
    SavedExpert(mlp).save(path)

    loaded = SavedExpert.load(path, device=CPU).mlp
    x = torch.randn(8, 4)
    assert torch.allclose(loaded.forward(x), mlp.forward(x), atol=1e-6)


def test_saved_expert_is_safetensors_not_pickle(tmp_path):
    path = tmp_path / "expert_0.pt"
    SavedExpert(AdvancedMLP(4, 6, 4, device=CPU)).save(path)
    # The artifact is a versioned safetensors payload, never a pickled module.
    assert S.peek(path).header[S.PAYLOAD_TYPE_FIELD] == S.PAYLOAD_SAVED_EXPERT


def test_saved_expert_preserves_bf16_dtype(tmp_path):
    mlp = AdvancedMLP(4, 6, 4, device=CPU).to(dtype=torch.bfloat16)
    path = tmp_path / "expert_0.pt"
    SavedExpert(mlp).save(path)
    assert SavedExpert.load(path, device=CPU).mlp.gate.weight.dtype == torch.bfloat16
PAIRS = [(3, 1), (0, 2), (5, 4)]  # deliberately unsorted to exercise orientation


@pytest.fixture
def path(tmp_path):
    return tmp_path / "artifact.safetensors"


# --------------------------------------------------------------------------- #
# Partition: order/orientation invariant.
# --------------------------------------------------------------------------- #
def test_from_pairs_preserves_order_and_orientation():
    partition = Partition.from_pairs(PAIRS, method="activation_correlation")
    assert partition.num_groups == len(PAIRS)
    for i, pair in enumerate(PAIRS):
        assert partition.groups[i].members == (pair[0], pair[1])
    assert partition.method == "activation_correlation"


def test_to_pairs_roundtrips():
    partition = Partition.from_pairs(PAIRS)
    assert partition.to_pairs() == PAIRS


def test_group_size():
    assert Group(members=(1, 2)).size == 2
    assert Group(members=(1, 2, 3)).size == 3


def test_to_pairs_rejects_non_pairwise_groups():
    partition = Partition(groups=(Group(members=(0, 1, 2)),))
    with pytest.raises(ValueError):
        partition.to_pairs()


# --------------------------------------------------------------------------- #
# Partition: S2 round-trip + legacy migration tie.
# --------------------------------------------------------------------------- #
def test_partition_roundtrips_through_serializer(path):
    partition = Partition.from_pairs(PAIRS, method="random")
    partition.save(path)
    loaded = Partition.load(path)
    assert loaded == partition
    assert loaded.to_pairs() == PAIRS
    assert S.load_header(path)[S.PAYLOAD_TYPE_FIELD] == "partition"


def test_partition_is_metadata_only_artifact(path):
    Partition.from_pairs(PAIRS).save(path)
    info = S.peek(path)
    assert info.tensors == {}


def test_partition_with_singleton_and_triple_roundtrips(path):
    """Groups of size 1 and 3 survive the save/load round-trip unchanged."""
    partition = Partition(
        groups=(Group(members=(0,)), Group(members=(1, 2, 3))), method="reap"
    )
    partition.save(path)
    loaded = Partition.load(path)
    assert loaded == partition
    assert [group.members for group in loaded.groups] == [(0,), (1, 2, 3)]
    assert [group.size for group in loaded.groups] == [1, 3]


def test_legacy_pairs_pickle_migrates_to_partition(tmp_path):
    legacy = tmp_path / "legacy_pairs.pt"
    torch.save(PAIRS, legacy)  # the old pipeline pickled the raw tuple list

    artifact = S.load_legacy(legacy)
    pairs = S.artifact_to_pairs(artifact)
    partition = Partition.from_pairs(pairs)

    for i, pair in enumerate(PAIRS):
        assert partition.groups[i].members == (pair[0], pair[1])
    assert partition.to_pairs() == PAIRS


# --------------------------------------------------------------------------- #
# GroupActivationIndex.
# --------------------------------------------------------------------------- #
def test_group_activation_index_roundtrips(path):
    index = GroupActivationIndex.from_lists([[0, 4, 7], [], [2, 3]], split="train")
    index.save(path)
    loaded = GroupActivationIndex.load(path)
    assert loaded == index
    assert loaded.to_lists() == [[0, 4, 7], [], [2, 3]]
    assert loaded.split == "train"
    assert loaded.num_groups == 3
    assert S.load_header(path)[S.PAYLOAD_TYPE_FIELD] == "group_activation_index"


def test_group_activation_index_from_lists_coerces_ints():
    index = GroupActivationIndex.from_lists([[torch.tensor(1), 2]])
    assert index.indices == ((1, 2),)
    assert all(isinstance(i, int) for i in index.indices[0])


# --------------------------------------------------------------------------- #
# CoActivationMatrix.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_coactivation_matrix_roundtrips(path, dtype):
    matrix = torch.randn(4, 4, dtype=dtype)
    art = CoActivationMatrix(matrix=matrix, split="validation", sample_size=256)
    art.save(path)
    loaded = CoActivationMatrix.load(path)

    assert torch.equal(loaded.matrix, matrix)
    assert loaded.matrix.dtype == dtype
    assert loaded.split == "validation"
    assert loaded.sample_size == 256
    assert loaded.num_experts == 4
    assert S.load_header(path)[S.PAYLOAD_TYPE_FIELD] == "coactivation_matrix"


def test_coactivation_matrix_optional_metadata(path):
    art = CoActivationMatrix(matrix=torch.zeros(3, 3))
    art.save(path)
    loaded = CoActivationMatrix.load(path)
    assert loaded.split is None
    assert loaded.sample_size is None
    assert loaded.num_experts == 3
