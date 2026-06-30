"""Data artifact classes for the pipeline.

Each stage writes typed objects (not bare tensors or pickled modules) using
versioned state_dict + JSON header serialization over safetensors.

Classes:
* Partition / Group -- grouping result: ordered list of expert-index groups plus method name.
  from_pairs preserves order and orientation: group i gets members (pairs[i][0], pairs[i][1]).
* GroupActivationIndex -- per-group lists of example indices that activate each group.
* CoActivationMatrix -- EXPERTS x EXPERTS co-activation matrix with split/sample metadata.

Library code only; never imports an experiment config.
"""

from dataclasses import dataclass
from typing import Optional

import torch

from common import serialization
from common.serialization import Artifact

# Payload type tags used during serialization.
PAYLOAD_PARTITION = "partition"
PAYLOAD_GROUP_ACTIVATION_INDEX = "group_activation_index"
PAYLOAD_COACTIVATION_MATRIX = "coactivation_matrix"

# Header key for the matrix tensor.
_MATRIX_TENSOR_KEY = "matrix"


@dataclass(frozen=True)
class Group:
    """Ordered tuple of expert indices forming one fused group.

    Member order is the orientation contract: member 0 reconstructs as out=0,
    member 1 as out=1. A pair group has size == 2.
    """

    members: tuple[int, ...]

    @property
    def size(self) -> int:
        return len(self.members)


@dataclass(frozen=True)
class Partition:
    """Ordered list of Groups plus the grouping method name.

    Group order defines the per-group file index; member order within each group
    defines the out=0/1 reconstruction orientation.
    """

    groups: tuple[Group, ...]
    method: Optional[str] = None

    @classmethod
    def from_pairs(cls, pairs, method: Optional[str] = None) -> "Partition":
        """Build a partition from a [(i, j), ...] list.

        Order- and orientation-preserving: group i gets members (pairs[i][0], pairs[i][1]).
        """
        groups = tuple(Group(members=(int(a), int(b))) for (a, b) in pairs)
        return cls(groups=groups, method=method)

    def to_pairs(self) -> list[tuple[int, int]]:
        """Recover the ``[(i, j), ...]`` list (only valid when every group is a pair)."""
        for index, group in enumerate(self.groups):
            if group.size != 2:
                raise ValueError(
                    f"to_pairs requires size-2 groups; group {index} has size {group.size}"
                )
        return [(group.members[0], group.members[1]) for group in self.groups]

    @property
    def num_groups(self) -> int:
        return len(self.groups)

    def to_artifact(self) -> Artifact:
        header = {
            "groups": [list(group.members) for group in self.groups],
            "method": self.method,
            "num_groups": self.num_groups,
        }
        return Artifact(tensors={}, header=header)

    @classmethod
    def from_artifact(cls, artifact: Artifact) -> "Partition":
        groups = tuple(
            Group(members=tuple(int(m) for m in members))
            for members in artifact.header.get("groups", [])
        )
        return cls(groups=groups, method=artifact.header.get("method"))

    def save(self, path) -> None:
        artifact = self.to_artifact()
        serialization.save(
            path, artifact.tensors, artifact.header, payload_type=PAYLOAD_PARTITION
        )

    @classmethod
    def load(cls, path) -> "Partition":
        return cls.from_artifact(serialization.load(path))


@dataclass(frozen=True)
class GroupActivationIndex:
    """Per-group lists of example indices that activate each group.

    indices[g] holds example indices that activated group g.
    Group order matches the Partition this was built from.
    """

    indices: tuple[tuple[int, ...], ...]
    split: Optional[str] = None

    @classmethod
    def from_lists(cls, lists, split: Optional[str] = None) -> "GroupActivationIndex":
        return cls(
            indices=tuple(tuple(int(i) for i in group) for group in lists),
            split=split,
        )

    def to_lists(self) -> list[list[int]]:
        return [list(group) for group in self.indices]

    @property
    def num_groups(self) -> int:
        return len(self.indices)

    def to_artifact(self) -> Artifact:
        header = {
            "indices": [list(group) for group in self.indices],
            "split": self.split,
            "num_groups": self.num_groups,
        }
        return Artifact(tensors={}, header=header)

    @classmethod
    def from_artifact(cls, artifact: Artifact) -> "GroupActivationIndex":
        return cls(
            indices=tuple(
                tuple(int(i) for i in group)
                for group in artifact.header.get("indices", [])
            ),
            split=artifact.header.get("split"),
        )

    def save(self, path) -> None:
        artifact = self.to_artifact()
        serialization.save(
            path,
            artifact.tensors,
            artifact.header,
            payload_type=PAYLOAD_GROUP_ACTIVATION_INDEX,
        )

    @classmethod
    def load(cls, path) -> "GroupActivationIndex":
        return cls.from_artifact(serialization.load(path))


@dataclass(frozen=True)
class CoActivationMatrix:
    """EXPERTS x EXPERTS co-activation matrix with capture metadata.

    Stores the data split and sample size so grouping provenance is self-contained.
    """

    matrix: torch.Tensor
    split: Optional[str] = None
    sample_size: Optional[int] = None

    @property
    def num_experts(self) -> int:
        return int(self.matrix.shape[0])

    def to_artifact(self) -> Artifact:
        header = {
            "split": self.split,
            "sample_size": self.sample_size,
            "num_experts": self.num_experts,
        }
        return Artifact(tensors={_MATRIX_TENSOR_KEY: self.matrix}, header=header)

    @classmethod
    def from_artifact(cls, artifact: Artifact) -> "CoActivationMatrix":
        header = artifact.header
        sample_size = header.get("sample_size")
        return cls(
            matrix=artifact.tensors[_MATRIX_TENSOR_KEY],
            split=header.get("split"),
            sample_size=None if sample_size is None else int(sample_size),
        )

    def save(self, path) -> None:
        artifact = self.to_artifact()
        serialization.save(
            path,
            artifact.tensors,
            artifact.header,
            payload_type=PAYLOAD_COACTIVATION_MATRIX,
        )

    @classmethod
    def load(cls, path) -> "CoActivationMatrix":
        return cls.from_artifact(serialization.load(path))


class SavedExpert:
    """A single extracted AdvancedMLP expert saved as a versioned safetensors artifact.

    Loads back forward-identically with dtype preserved; never pickled.
    """

    def __init__(self, mlp):
        self.mlp = mlp

    def save(self, path, *, provenance: Optional[dict] = None) -> None:
        artifact = serialization.mlp_to_artifact(self.mlp)
        serialization.save(
            path,
            artifact.tensors,
            artifact.header,
            payload_type=serialization.PAYLOAD_SAVED_EXPERT,
            provenance=provenance,
        )

    @classmethod
    def load(cls, path, device: Optional[torch.device] = None) -> "SavedExpert":
        return cls(serialization.artifact_to_mlp(serialization.load(path), device=device))
