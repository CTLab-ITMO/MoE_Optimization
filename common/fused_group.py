"""Runtime reconstruction contract for a fused group of experts.

A fused group is the result of merging ``k`` experts into one compact module.
At inference the layer must be able to recover *each original member's* output
from that compact module. The old pipeline did this with
:meth:`AdvancedMLP.forward_unzip` and a positional ``out=0|1`` argument baked
into :class:`~common.fusion_layer.FusionLayer`. That hard-codes ``k == 2``.

This module defines the contract the layer depends on instead:

* :class:`FusedGroup` -- abstract, ``k``-agnostic: ``group_size`` plus
  ``reconstruct(x, member) -> Tensor`` (the output of member ``member`` for
  input ``x``). A ``k > 2`` group can implement this without touching the layer.
* :class:`MLPFusedGroup` -- the concrete ``k == 2`` implementation backing the
  current pipeline. Its :meth:`reconstruct` *is* one implementation of the
  contract: it delegates straight to :meth:`AdvancedMLP.forward_unzip`, so it is
  numerically identical to the legacy path.
* :class:`FusedExpert` -- the on-disk artifact (§2.4) for a fused group. It
  serializes the *reconstruction contract* (group size + reconstruction kind),
  not just the weights, on top of the versioned ``state_dict`` + header
  serializer (§3.5). It never pickles an ``nn.Module``.

:class:`~common.fusion_layer.FusionLayer` now reconstructs through
:meth:`reconstruct` (wrapping each fused expert in an :class:`MLPFusedGroup`)
instead of the baked-in ``forward_unzip(out=)``, and the least-squares adapter
refits its unmerge matrix through the public
:meth:`~common.mlp.AdvancedMLP.refit_unmerge` rather than poking ``.U``/``.down``
from the outside -- both wired in S5 and proven equivalent to ``forward_unzip``.

Library code only -- this module never imports an experiment ``config``.
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch

from common import serialization
from common.mlp import AdvancedMLP
from common.serialization import Artifact, SerializationError

# Reconstruction-kind tag stored in a FusedExpert header. It names the
# :class:`FusedGroup` implementation needed to rebuild the runtime group, so a
# future ``k > 2`` (or least-squares ``refit_unmerge``) reconstruction is an
# additive new tag rather than a breaking change.
ZIP_UNMERGE_RECONSTRUCTION = "zip_unmerge"

# Header fields the FusedExpert artifact owns on top of the AdvancedMLP header.
GROUP_SIZE_FIELD = "group_size"
RECONSTRUCTION_FIELD = "reconstruction"

# Payload-type tag for a fused-group artifact.
PAYLOAD_FUSED_EXPERT = "fused_expert"

# The zip/``build_MU`` unmerge is strictly a pairwise (k == 2) construction; its
# generalization is left to other researchers (§3.1).
_ZIP_UNMERGE_GROUP_SIZE = 2


class FusedGroup(ABC):
    """Abstract, ``k``-agnostic runtime contract for a fused group of experts.

    A concrete group reconstructs the output of any one of its ``group_size``
    members for a given input. :class:`~common.fusion_layer.FusionLayer` depends
    on this contract, not on a fixed ``out=0|1`` argument, so a ``k > 2`` group
    works without changing the layer.
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
    """Pairwise (``k == 2``) fused group backed by :meth:`AdvancedMLP.forward_unzip`.

    The reconstruction of member ``m`` is exactly ``mlp.forward_unzip(x, out=m)``
    -- the zip/``build_MU`` unmerge -- so this group is numerically identical to
    the legacy :class:`~common.fusion_layer.FusionLayer` path. The member order
    is the orientation contract: member ``0`` is the first original expert,
    member ``1`` the second (``out=(0 if expert_idx == pair[0] else 1)``).
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
    """Serializable artifact (§2.4) for one fused group of ``k`` experts.

    Wraps the fused :class:`AdvancedMLP` together with the reconstruction
    contract metadata (``group_size`` + reconstruction kind) so a loader can
    rebuild the *runtime* :class:`FusedGroup`, not just the weights. Persisted
    through the versioned ``state_dict`` + header serializer (§3.5): never a
    pickled module.
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
        """Encode as a ``state_dict`` + header :class:`~common.serialization.Artifact`.

        Reuses :func:`common.serialization.mlp_to_artifact` for the weights and
        shape/activation header, and adds the reconstruction-contract fields.
        """
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

    def save(self, path) -> None:
        """Persist to ``path`` as a versioned safetensors artifact."""
        artifact = self.to_artifact()
        serialization.save(
            path,
            artifact.tensors,
            artifact.header,
            payload_type=PAYLOAD_FUSED_EXPERT,
        )

    @classmethod
    def load(cls, path, device: Optional[torch.device] = None) -> "FusedExpert":
        """Load a fused-expert artifact and rebuild it."""
        return cls.from_artifact(serialization.load(path), device=device)
