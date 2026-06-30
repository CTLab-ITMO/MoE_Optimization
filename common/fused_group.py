"""Runtime reconstruction contract for a fused group of experts.

Merging k experts produces a compact module; at inference the layer must
recover each original member's output from it. This module defines:

* :class:`FusedGroup` -- k-agnostic abstract contract: ``group_size`` and
  ``reconstruct(x, member) -> Tensor``.
* :class:`MLPFusedGroup` -- concrete k==2 implementation via
  :meth:`AdvancedMLP.forward_unzip`.
* :class:`FusedExpert` -- on-disk artifact storing weights plus reconstruction
  contract metadata (group size + kind). Never pickles an nn.Module.

Library code only -- this module never imports an experiment config.
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch

from common import serialization
from common.mlp import AdvancedMLP
from common.serialization import Artifact, SerializationError

# Tag stored in the FusedExpert header naming which FusedGroup implementation
# to use. Adding a new tag handles future k>2 variants without breaking changes.
ZIP_UNMERGE_RECONSTRUCTION = "zip_unmerge"

# Header fields the FusedExpert artifact owns on top of the AdvancedMLP header.
GROUP_SIZE_FIELD = "group_size"
RECONSTRUCTION_FIELD = "reconstruction"

# Payload-type tag for a fused-group artifact.
PAYLOAD_FUSED_EXPERT = "fused_expert"

# zip/build_MU unmerge is strictly pairwise (k == 2).
_ZIP_UNMERGE_GROUP_SIZE = 2


class FusedGroup(ABC):
    """Abstract k-agnostic contract: reconstruct any member's output from the fused module.

    FusionLayer depends on this, not a fixed out=0|1 arg, so k>2 works without changing the layer.
    """

    @property
    @abstractmethod
    def group_size(self) -> int:
        """Number of original experts (``k``) fused into this group."""

    @abstractmethod
    def reconstruct(self, x: torch.Tensor, member: int) -> torch.Tensor:
        """Reconstruct the output of member ``member`` (``0 <= member < k``) for ``x``."""

    def _validate_member(self, member: int) -> None:
        if not isinstance(member, int) or isinstance(member, bool):
            raise TypeError(f"member must be an int, got {type(member).__name__}")
        if not 0 <= member < self.group_size:
            raise IndexError(
                f"member {member} out of range for group_size {self.group_size}"
            )


class MLPFusedGroup(FusedGroup):
    """Pairwise (k==2) fused group: reconstruct(x, m) == mlp.forward_unzip(x, out=m).

    Member 0 = first original expert, member 1 = second.
    """

    reconstruction = ZIP_UNMERGE_RECONSTRUCTION

    def __init__(self, mlp: AdvancedMLP):
        if not isinstance(mlp, AdvancedMLP):
            raise TypeError(
                f"MLPFusedGroup wraps an AdvancedMLP, got {type(mlp).__name__}"
            )
        self._mlp = mlp

    @property
    def mlp(self) -> AdvancedMLP:
        """The fused :class:`AdvancedMLP` backing this group."""
        return self._mlp

    @property
    def group_size(self) -> int:
        return _ZIP_UNMERGE_GROUP_SIZE

    def reconstruct(self, x: torch.Tensor, member: int) -> torch.Tensor:
        self._validate_member(member)
        return self._mlp.forward_unzip(x, out=member)


class FusedExpert:
    """On-disk artifact for one fused group.

    Stores the AdvancedMLP weights plus group_size and reconstruction kind so
    the loader can rebuild the runtime FusedGroup. Never pickled.
    """

    def __init__(
        self,
        mlp: AdvancedMLP,
        *,
        group_size: int = _ZIP_UNMERGE_GROUP_SIZE,
        reconstruction: str = ZIP_UNMERGE_RECONSTRUCTION,
    ):
        if reconstruction != ZIP_UNMERGE_RECONSTRUCTION:
            raise SerializationError(
                f"unknown reconstruction kind {reconstruction!r}; "
                f"only {ZIP_UNMERGE_RECONSTRUCTION!r} is supported"
            )
        if group_size != _ZIP_UNMERGE_GROUP_SIZE:
            raise SerializationError(
                f"reconstruction {reconstruction!r} supports only group_size "
                f"{_ZIP_UNMERGE_GROUP_SIZE}, got {group_size}"
            )
        self.mlp = mlp
        self.group_size = int(group_size)
        self.reconstruction = reconstruction

    @classmethod
    def from_group(cls, group: MLPFusedGroup) -> "FusedExpert":
        """Build a serializable expert from a runtime :class:`MLPFusedGroup`."""
        return cls(
            group.mlp,
            group_size=group.group_size,
            reconstruction=group.reconstruction,
        )

    def to_group(self) -> MLPFusedGroup:
        """Materialize the runtime :class:`FusedGroup` for inference."""
        return MLPFusedGroup(self.mlp)

    def to_artifact(self) -> Artifact:
        """Encode to a state_dict + header Artifact, adding reconstruction-contract fields."""
        base = serialization.mlp_to_artifact(self.mlp)
        header = {
            **base.header,
            GROUP_SIZE_FIELD: self.group_size,
            RECONSTRUCTION_FIELD: self.reconstruction,
        }
        return Artifact(tensors=base.tensors, header=header)

    @classmethod
    def from_artifact(
        cls, artifact: Artifact, device: Optional[torch.device] = None
    ) -> "FusedExpert":
        """Rebuild a :class:`FusedExpert` forward-identically from an artifact."""
        header = artifact.header
        reconstruction = header.get(RECONSTRUCTION_FIELD, ZIP_UNMERGE_RECONSTRUCTION)
        group_size = int(header.get(GROUP_SIZE_FIELD, _ZIP_UNMERGE_GROUP_SIZE))
        mlp = serialization.artifact_to_mlp(artifact, device=device)
        return cls(mlp, group_size=group_size, reconstruction=reconstruction)

    def save(self, path, *, provenance: Optional[dict] = None) -> None:
        """Save to path as a versioned safetensors file.

        provenance is stored in the header; the fusion step uses it to skip
        already-fresh work items on resume.
        """
        artifact = self.to_artifact()
        serialization.save(
            path,
            artifact.tensors,
            artifact.header,
            payload_type=PAYLOAD_FUSED_EXPERT,
            provenance=provenance,
        )

    @classmethod
    def load(cls, path, device: Optional[torch.device] = None) -> "FusedExpert":
        """Load a fused-expert artifact and rebuild it."""
        return cls.from_artifact(serialization.load(path), device=device)
