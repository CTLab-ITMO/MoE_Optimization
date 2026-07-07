"""Model-family adapters for expert extraction and the shared-expert pathway.

Isolates two places that know the concrete MoE module layout, so supporting a
new model family only requires a new subclass.

* ``ExpertAdapter`` -- wraps each routed expert as an ``AdvancedMLP``.
  Handles two layouts: a ModuleList of per-expert modules (shares projections
  via ``turn_in_mlp``) and a combined container with stacked weight tensors
  (clones per-expert slices). No dtype cast -- the caller casts.

* ``SharedExpertAdapter`` -- DI adapter for the shared/residual-expert pathway.
  ``extract(mlp)`` pulls the shared-expert submodules from a live MoE ``mlp`` for
  stage 1 to persist (or ``None`` for families without one, e.g. Mixtral);
  ``build(extracted)`` assembles the runtime contribution module for the fusion
  layer. ``Qwen15SharedExpertAdapter`` builds a ``Qwen15SharedExpert`` (applies
  ``sigmoid(gate(x)) * shared(x)``); ``NoSharedExpertAdapter`` is the fallback for
  families with no shared expert -- ``extract`` returns ``None`` and ``build``
  yields a zeros-contributing module, so the layer's ``+ shared`` term vanishes.

Library code only -- never imports an experiment ``config``.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
from torch import nn

from common.mlp import AdvancedMLP
from common.mlp_utils import turn_in_mlp


def _require_float_weight(weight, what: str) -> torch.Tensor:
    """Raise TypeError if ``weight`` is not a floating-point tensor.

    Quantized experts may expose None or an integer weight; this catches that
    early with a clear message instead of a cryptic downstream error.
    """
    if not torch.is_tensor(weight) or not torch.is_floating_point(weight):
        raise TypeError(
            f"quantized experts are not supported: {what} has no floating-point "
            f"weight (got {type(weight).__name__})"
        )
    return weight


# --------------------------------------------------------------------------- #
# Expert extraction adapter.
# --------------------------------------------------------------------------- #
class ExpertAdapter(ABC):
    """Abstract base: extract MoE experts as ``AdvancedMLP`` objects.

    Subclasses implement ``count`` (number of experts in the container) and
    ``extract`` (build an ``AdvancedMLP`` for one expert; no dtype cast).
    """

    @abstractmethod
    def count(self, experts) -> int:
        """Return the number of experts held by ``experts``."""
        raise NotImplementedError()

    @abstractmethod
    def extract(self, experts, index: int, *, device) -> AdvancedMLP:
        """Return the expert at ``index`` wrapped as an ``AdvancedMLP`` on ``device``."""
        raise NotImplementedError()


class Qwen15ExpertAdapter(ExpertAdapter):
    """Expert extraction for Qwen1.5-MoE, supporting two module layouts.

    Auto-detects the layout: iterable (ModuleList of per-expert modules) vs.
    a single combined container (raises TypeError on iteration).
    """

    @staticmethod
    def _is_module_list(experts) -> bool:
        """Return True if ``experts`` is a ModuleList, False for the combined layout.

        Detects by calling ``iter(experts)``: a non-iterable combined container
        raises TypeError, selecting the combined branch.
        """
        try:
            iter(experts)
        except TypeError:
            return False
        return True

    def count(self, experts) -> int:
        if self._is_module_list(experts):
            return len(experts)
        return experts.num_experts

    def extract(self, experts, index: int, *, device) -> AdvancedMLP:
        if self._is_module_list(experts):
            return self._extract_module_list(experts, index, device)
        return self._extract_combined(experts, index, device)

    def _extract_module_list(self, experts, index: int, device) -> AdvancedMLP:
        """ModuleList branch: wrap the expert via ``turn_in_mlp`` (shares projections).

        The returned ``AdvancedMLP`` reuses gate/up/down projections directly
        (no clone). Validates weights first to catch quantized experts early.
        """
        expert = experts[index]
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            proj = getattr(expert, proj_name, None)
            _require_float_weight(
                getattr(proj, "weight", None), f"expert {index} {proj_name}.weight"
            )
        return turn_in_mlp(expert, device=device)

    def _extract_combined(self, experts, index: int, device) -> AdvancedMLP:
        """Combined branch: build a fresh ``AdvancedMLP`` with cloned weight slices.

        The container exposes stacked ``gate_up_proj`` [E, 2*intermediate, hidden]
        and ``down_proj`` [E, hidden, intermediate]. Gate and up weights are sliced
        at ``intermediate_dim`` and cloned (not shared).
        """
        gate_up_proj = _require_float_weight(
            getattr(experts, "gate_up_proj", None), "experts.gate_up_proj"
        )
        down_proj = _require_float_weight(
            getattr(experts, "down_proj", None), "experts.down_proj"
        )
        intermediate_dim = experts.intermediate_dim
        new_mlp = AdvancedMLP(
            experts.hidden_dim,
            intermediate_dim,
            experts.hidden_dim,
            device=device,
        )
        new_mlp.gate.weight = nn.Parameter(
            gate_up_proj[index, :intermediate_dim].detach().clone()
        )
        new_mlp.up.weight = nn.Parameter(
            gate_up_proj[index, intermediate_dim:].detach().clone()
        )
        new_mlp.down.weight = nn.Parameter(down_proj[index].detach().clone())
        return new_mlp


# --------------------------------------------------------------------------- #
# Shared-expert runtime module (built by SharedExpertAdapter.build).
# --------------------------------------------------------------------------- #
class Qwen15SharedExpert(torch.nn.Module):
    """Sigmoid-gated shared-expert runtime for Qwen1.5-MoE.

    Computes ``sigmoid(gate(x)) * shared(x)``. When both submodules are
    ``None`` (no shared expert, e.g. Mixtral / Qwen3), returns zeros instead.
    Check ``has_shared`` to test which mode is active. This is an ``nn.Module``
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

    @property
    def has_shared(self) -> bool:
        """Whether this layer has a real shared expert (vs the zeros fallback)."""
        return self.shared_expert is not None


# --------------------------------------------------------------------------- #
# Shared-expert DI adapter: extraction (stage 1) + runtime build (fusion layer).
# --------------------------------------------------------------------------- #
class SharedExpertAdapter(ABC):
    """Abstract base: the shared/residual-expert pathway of a model family.

    Two family-specific responsibilities decouple the stages from the concrete
    MoE layout:

    * ``extract(mlp)`` -- pull ``(shared_expert, shared_expert_gate)`` from a live
      MoE ``mlp`` module for stage 1 to persist, or ``None`` for families without
      a shared expert (e.g. Mixtral).
    * ``build(extracted)`` -- assemble the runtime contribution ``nn.Module`` from
      the (loaded) submodules for the fusion layer; given ``None`` it returns a
      zeros-contributing module so the layer's ``+ shared`` term is a no-op.
    """

    @abstractmethod
    def extract(self, mlp) -> Optional[Tuple[nn.Module, nn.Module]]:
        """Return ``(shared_expert, shared_expert_gate)`` from ``mlp``, or ``None``."""
        raise NotImplementedError()

    @abstractmethod
    def build(self, extracted: Optional[Tuple[nn.Module, nn.Module]]) -> nn.Module:
        """Build the runtime shared-expert module from ``extracted`` (or a zeros module)."""
        raise NotImplementedError()


class Qwen15SharedExpertAdapter(SharedExpertAdapter):
    """Shared-expert adapter for Qwen1.5-MoE (sigmoid-gated shared expert).

    Extracts the ``shared_expert`` / ``shared_expert_gate`` submodules the Qwen1.5
    MoE block exposes and builds a :class:`Qwen15SharedExpert`. ``extract`` returns
    ``None`` if the module has no ``shared_expert`` attribute, degrading to the
    zeros fallback rather than raising.
    """

    def extract(self, mlp) -> Optional[Tuple[nn.Module, nn.Module]]:
        shared_expert = getattr(mlp, "shared_expert", None)
        if shared_expert is None:
            return None
        return shared_expert, getattr(mlp, "shared_expert_gate", None)

    def build(self, extracted: Optional[Tuple[nn.Module, nn.Module]]) -> nn.Module:
        if extracted is None:
            return Qwen15SharedExpert(None, None)
        return Qwen15SharedExpert(*extracted)


class NoSharedExpertAdapter(SharedExpertAdapter):
    """Shared-expert adapter for families without one (e.g. Mixtral / Qwen3-MoE).

    ``extract`` always returns ``None`` (nothing to persist) and ``build`` always
    yields a zeros-contributing module, so the fusion layer's shared term vanishes.
    """

    def extract(self, mlp) -> Optional[Tuple[nn.Module, nn.Module]]:
        return None

    def build(self, extracted: Optional[Tuple[nn.Module, nn.Module]]) -> nn.Module:
        return Qwen15SharedExpert(None, None)
