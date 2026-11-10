"""Model-family adapter for the shared-expert pathway of a fused MoE layer.

``SharedExpertAdapter`` -- DI adapter: ``build(extracted)`` assembles the runtime
shared-expert contribution module for the fusion layer from the loaded
``(shared_expert, shared_expert_gate)`` submodules, or from ``None`` for a layer
without a shared expert. ``Qwen15SharedExpertAdapter`` builds a ``Qwen15SharedExpert``
(applies ``sigmoid(gate(x)) * shared(x)``, or contributes zeros given ``None``).

Library code only -- never imports an experiment ``config``.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
from torch import nn


# --------------------------------------------------------------------------- #
# Shared-expert runtime module (built by SharedExpertAdapter.build).
# --------------------------------------------------------------------------- #
class Qwen15SharedExpert(torch.nn.Module):
    """Sigmoid-gated shared-expert runtime for Qwen1.5-MoE.

    Computes ``sigmoid(gate(x)) * shared(x)``. When both submodules are
    ``None`` (no shared expert), returns zeros instead. This is an ``nn.Module``
    so ``.to(...)`` moves the registered submodules with the fusion layer.
    """

    def __init__(
        self,
        shared_expert: Optional[nn.Module],
        shared_expert_gate: Optional[nn.Module],
    ):
        super().__init__()
        self.shared_expert = shared_expert
        self.shared_expert_gate = shared_expert_gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.shared_expert is None:
            return torch.zeros_like(x)
        return torch.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)


# --------------------------------------------------------------------------- #
# Shared-expert DI adapter: runtime build for the fusion layer.
# --------------------------------------------------------------------------- #
class SharedExpertAdapter(ABC):
    """Abstract base: the shared/residual-expert pathway of a model family.

    ``build(extracted)`` assembles the runtime contribution ``nn.Module`` from
    the loaded submodules for the fusion layer; given ``None`` it returns a
    zeros-contributing module so the layer's ``+ shared`` term is a no-op.
    """

    @abstractmethod
    def build(self, extracted: Optional[Tuple[nn.Module, nn.Module]]) -> nn.Module:
        """Build the runtime shared-expert module from ``extracted`` (or a zeros module)."""
        raise NotImplementedError()


class Qwen15SharedExpertAdapter(SharedExpertAdapter):
    """Shared-expert adapter for Qwen1.5-MoE (sigmoid-gated shared expert)."""

    def build(self, extracted: Optional[Tuple[nn.Module, nn.Module]]) -> nn.Module:
        if extracted is None:
            return Qwen15SharedExpert(None, None)
        return Qwen15SharedExpert(*extracted)
