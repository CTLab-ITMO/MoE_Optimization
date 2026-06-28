"""Meaningful stored-data types for the pipeline (§2.4).

Every artifact a stage writes is an object of a class with its own interface and
metadata, persisted through the versioned ``state_dict`` + header serializer
(§3.5) -- never a bare list/tensor and never a pickled module. This module holds
the *data* artifacts; the fused-group runtime contract lives in
:mod:`common.fused_group`.

Classes:

* :class:`Partition` / :class:`Group` -- a grouping result (list of groups, each
  a collection of expert indices) plus the grouping-method metadata. Replaces
  the raw ``pairs/N.pt`` tuple list. :meth:`Partition.from_pairs` is **order-
  and orientation-preserving**: ``from_pairs(pairs).groups[i].members ==
  (pairs[i][0], pairs[i][1])`` for all ``i`` -- the silent-break invariant that
  keeps the ``i.pt`` index and the ``out=0/1`` member orientation in sync.
* :class:`GroupActivationIndex` -- per-group lists of the example indices that
  activate each group. Replaces the raw per-pair lists of the old stage 5.
* :class:`CoActivationMatrix` -- an ``EXPERTS x EXPERTS`` co-activation matrix
  plus metadata (split, sample size).

Library code only -- this module never imports an experiment ``config``.
"""

from dataclasses import dataclass
from typing import Optional

import torch

from common import serialization
from common.serialization import Artifact

# Payload-type tags for the data artifacts.
PAYLOAD_PARTITION = "partition"
PAYLOAD_GROUP_ACTIVATION_INDEX = "group_activation_index"
PAYLOAD_COACTIVATION_MATRIX = "coactivation_matrix"

# Header field holding the co-activation matrix tensor.
_MATRIX_TENSOR_KEY = "matrix"


@dataclass(frozen=True)
class Group:
    """One group of fused experts: an ordered tuple of expert indices.

    The member order is the orientation contract (member ``0`` reconstructs as
    ``out=0``, member ``1`` as ``out=1``); a pair is ``size == 2``.
    """

    members: tuple[int, ...]

    @property
    def size(self) -> int:
        return len(self.members)


@dataclass(frozen=True)
class Partition:
    """A grouping result: an ordered list of :class:`Group` + method metadata.

    The group order is the ``i.pt`` index contract; the per-group member order is
    the ``out=0/1`` orientation contract. Replaces the raw ``pairs/N.pt`` list.
    """

    groups: tuple[Group, ...]
    method: Optional[str] = None

    @classmethod
    def from_pairs(cls, pairs, method: Optional[str] = None) -> "Partition":
        """Build a (pairwise) partition from a ``[(i, j), ...]`` list.

        Order- and orientation-preserving by construction: group ``i`` is
        ``(pairs[i][0], pairs[i][1])`` -- the silent-break invariant.
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

    ``indices[g]`` is the list of example indices that activated group ``g``.
    Replaces the raw per-pair index lists of the old stage 5; the group order
    matches the :class:`Partition` it was built against.
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
    """An ``EXPERTS x EXPERTS`` co-activation matrix plus capture metadata.

    Carries the split it was captured on and the sample size, so a reader knows
    the provenance of the grouping signal without re-deriving it.
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
