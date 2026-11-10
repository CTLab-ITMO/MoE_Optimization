"""Runtime reconstruction contract for a fused group of experts.

Merging k experts produces a compact module; at inference the layer must
recover each original member's output from it. This module defines:

* :class:`FusedGroup` -- k-agnostic abstract contract: ``group_size`` and
  ``reconstruct(x, member) -> Tensor``.
* :class:`MLPFusedGroup` -- concrete k==2 implementation via
  :meth:`AdvancedMLP.forward_unzip`.

Library code only -- this module never imports an experiment config.
"""

from abc import ABC, abstractmethod

import torch

from common.mlp import AdvancedMLP

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

    def __init__(self, mlp: AdvancedMLP):
        if not isinstance(mlp, AdvancedMLP):
            raise TypeError(
                f"MLPFusedGroup wraps an AdvancedMLP, got {type(mlp).__name__}"
            )
        self._mlp = mlp

    @property
    def group_size(self) -> int:
        return _ZIP_UNMERGE_GROUP_SIZE

    def reconstruct(self, x: torch.Tensor, member: int) -> torch.Tensor:
        self._validate_member(member)
        return self._mlp.forward_unzip(x, out=member)
