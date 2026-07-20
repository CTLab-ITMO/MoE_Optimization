"""Runtime reconstruction contract for a fused group of experts.

Merging k experts produces a compact module; at inference the layer must
recover each original member's output from it. This module defines:

* :class:`FusedGroup` -- k-agnostic abstract contract: ``group_size`` and
  ``reconstruct(x, member) -> Tensor``.
* :class:`MLPFusedGroup` -- pairwise (k==2) implementation via
  :meth:`AdvancedMLP.forward_unzip`.
* :class:`IdentityFusedGroup` -- k-agnostic (k>=1) implementation returning
  the fused output as-is for every member.
* :data:`RECONSTRUCTION_REGISTRY` -- reconstruction kind ->
  :class:`ReconstructionSpec` (group-size contract + runtime builder); the
  :class:`FusedExpert` validator and loader read it, so a new reconstruction
  kind is added by registering a spec, not by editing the artifact class.
* :class:`FusedExpert` -- on-disk artifact storing weights plus reconstruction
  contract metadata (group size + kind). Never pickles an nn.Module.

Library code only -- this module never imports an experiment config.
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional

import torch

from common import serialization
from common.mlp import AdvancedMLP
from common.registry import Registry
from common.serialization import Artifact, SerializationError

# Tags stored in the FusedExpert header naming which FusedGroup implementation
# to use; RECONSTRUCTION_REGISTRY below maps each tag to its spec.
ZIP_UNMERGE_RECONSTRUCTION = "zip_unmerge"
IDENTITY_RECONSTRUCTION = "identity"

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


class IdentityFusedGroup(FusedGroup):
    """K-agnostic (k>=1) fused group: every member reconstructs to the fused
    output as-is (``reconstruct(x, m) == mlp(x)``).

    Covers methods whose result is a plain expert with no unmerge state, e.g.
    Pruning keeps one member and a weighted average serves all members.
    """

    reconstruction = IDENTITY_RECONSTRUCTION

    def __init__(self, mlp: AdvancedMLP, group_size: int):
        if not isinstance(mlp, AdvancedMLP):
            raise TypeError(
                f"IdentityFusedGroup wraps an AdvancedMLP, got {type(mlp).__name__}"
            )
        if (
            not isinstance(group_size, int)
            or isinstance(group_size, bool)
            or group_size < 1
        ):
            raise ValueError(f"group_size must be an int >= 1, got {group_size!r}")
        self._mlp = mlp
        self._group_size = group_size

    @property
    def mlp(self) -> AdvancedMLP:
        """The fused :class:`AdvancedMLP` backing this group."""
        return self._mlp

    @property
    def group_size(self) -> int:
        return self._group_size

    def reconstruct(self, x: torch.Tensor, member: int) -> torch.Tensor:
        self._validate_member(member)
        return self._mlp.forward(x)


def _identity_reconstruct_all(mlp, group_size: int, x: torch.Tensor) -> torch.Tensor:
    """Concatenated identity reconstruction: the fused output tiled ``group_size``
    times along the last axis (a no-op for ``group_size == 1``)."""
    y = mlp.forward(x)
    return y.repeat(*([1] * (y.dim() - 1) + [group_size]))


class ReconstructionSpec:
    """Group-size contract and runtime builder for one reconstruction kind."""

    def __init__(
        self,
        kind: str,
        build: Callable[[AdvancedMLP, int], FusedGroup],
        *,
        min_group_size: int = 1,
        max_group_size: Optional[int] = None,
        reconstruct_all: Optional[Callable[[AdvancedMLP, int, torch.Tensor], torch.Tensor]] = None,
    ):
        self.kind = kind
        self._build = build
        self.min_group_size = min_group_size
        self.max_group_size = max_group_size
        # Concatenated-over-all-members reconstruction (mlp, group_size, x) -> Tensor.
        # Duck-typed on the mlp; no isinstance.
        self.reconstruct_all = reconstruct_all

    def accepts(self, group_size) -> bool:
        """Whether ``group_size`` fits this kind (non-raising :meth:`validate_group_size`)."""
        try:
            self.validate_group_size(group_size)
            return True
        except SerializationError:
            return False

    def validate_group_size(self, group_size) -> None:
        """Raise :class:`SerializationError` unless ``group_size`` fits this kind."""
        valid = (
            isinstance(group_size, int)
            and not isinstance(group_size, bool)
            and group_size >= self.min_group_size
            and (self.max_group_size is None or group_size <= self.max_group_size)
        )
        if not valid:
            if self.min_group_size == self.max_group_size:
                bound = f"exactly {self.min_group_size}"
            elif self.max_group_size is None:
                bound = f">= {self.min_group_size}"
            else:
                bound = f"{self.min_group_size}..{self.max_group_size}"
            raise SerializationError(
                f"reconstruction {self.kind!r} supports only group_size "
                f"{bound}, got {group_size!r}"
            )

    def build(self, mlp: AdvancedMLP, group_size: int) -> FusedGroup:
        """Materialize the runtime :class:`FusedGroup` for stored weights."""
        return self._build(mlp, group_size)


# Reconstruction kind -> spec. The FusedExpert validator and to_group() read
# this; register a new spec to support a new reconstruction kind.
RECONSTRUCTION_REGISTRY: Registry = Registry("reconstruction")
RECONSTRUCTION_REGISTRY.register(
    ZIP_UNMERGE_RECONSTRUCTION,
    ReconstructionSpec(
        ZIP_UNMERGE_RECONSTRUCTION,
        lambda mlp, group_size: MLPFusedGroup(mlp),
        min_group_size=_ZIP_UNMERGE_GROUP_SIZE,
        max_group_size=_ZIP_UNMERGE_GROUP_SIZE,
        # Single whole-pair call -- the historical k=2 path, byte-identical.
        reconstruct_all=lambda mlp, gs, x: mlp.forward_unzip(x),
    ),
)
RECONSTRUCTION_REGISTRY.register(
    IDENTITY_RECONSTRUCTION,
    ReconstructionSpec(
        IDENTITY_RECONSTRUCTION,
        IdentityFusedGroup,
        reconstruct_all=_identity_reconstruct_all,
    ),
)


def default_reconstruction_for_size(group_size: int) -> str:
    """First registered reconstruction kind (in registration order) accepting ``group_size``.

    zip_unmerge (exactly 2) is registered first and identity (>= 1) second, so
    ``group_size == 2`` resolves to zip_unmerge and any other size to identity --
    but only via each spec's :meth:`ReconstructionSpec.accepts`, never a literal.
    """
    for kind in RECONSTRUCTION_REGISTRY.names():
        if RECONSTRUCTION_REGISTRY[kind].accepts(group_size):
            return kind
    raise SerializationError(
        f"no registered reconstruction kind accepts group_size {group_size!r}; "
        f"registered: {RECONSTRUCTION_REGISTRY.names()}"
    )


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
        if reconstruction not in RECONSTRUCTION_REGISTRY:
            raise SerializationError(
                f"unknown reconstruction kind {reconstruction!r}; "
                f"registered: {RECONSTRUCTION_REGISTRY.names()}"
            )
        RECONSTRUCTION_REGISTRY[reconstruction].validate_group_size(group_size)
        self.mlp = mlp
        self.group_size = int(group_size)
        self.reconstruction = reconstruction

    @classmethod
    def from_group(cls, group: FusedGroup) -> "FusedExpert":
        """Build a serializable expert from a runtime :class:`FusedGroup`.

        The group must expose ``mlp`` and ``reconstruction`` on top of the
        abstract contract (both in-tree implementations do).
        """
        return cls(
            group.mlp,
            group_size=group.group_size,
            reconstruction=group.reconstruction,
        )

    @classmethod
    def from_result(cls, mlp, method, group_size: int) -> "FusedExpert":
        """Build a fused expert, choosing the reconstruction kind the method declares.

        The reconstruction-kind is declared by the method (``method.reconstruction``).
        If the method does not set it, or the declared kind is incompatible with
        this ``group_size`` (e.g. zip_unmerge for a singleton), the size default
        (:func:`default_reconstruction_for_size`) is used. ``method=None`` -- the
        pseudo-methods ``Best_*`` -- also falls back to the size default.
        """
        kind = getattr(method, "reconstruction", None)
        if kind is None or not RECONSTRUCTION_REGISTRY[kind].accepts(group_size):
            kind = default_reconstruction_for_size(group_size)
        return cls(mlp, group_size=group_size, reconstruction=kind)

    def to_group(self) -> FusedGroup:
        """Materialize the runtime :class:`FusedGroup` for inference."""
        return RECONSTRUCTION_REGISTRY[self.reconstruction].build(
            self.mlp, self.group_size
        )

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
