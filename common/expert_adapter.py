"""Model-family adapters for expert extraction and the shared-expert pathway.

Two pieces of the pipeline reach directly into a concrete MoE module layout and
hard-code Qwen1.5 assumptions; this module turns each into a model-family
adapter so a new family is a new subclass rather than an edit to a stage.

* :class:`ExpertAdapter` -- factors out the inline expert extraction that wraps
  each routed expert into an :class:`~common.mlp.AdvancedMLP`, picking one of two
  layouts via a ``try/except TypeError`` on ``for i, expert in enumerate(experts)``:

    - the **ModuleList** layout (one module per expert exposing
      ``.gate_proj`` / ``.up_proj`` / ``.down_proj``), handled by
      :func:`common.mlp_utils.turn_in_mlp`, which *shares* the projection
      modules; and
    - the **combined** layout (a single container exposing ``num_experts`` /
      ``hidden_dim`` / ``intermediate_dim`` and the stacked ``gate_up_proj`` /
      ``down_proj`` tensors), which builds a fresh ``AdvancedMLP`` and *clones*
      the per-expert weight slices.

  :class:`Qwen15ExpertAdapter` implements both branches (matching ops, dims,
  dtypes, orientation, sharing-vs-cloning). The caller casts to the model dtype
  afterwards, so :meth:`~Qwen15ExpertAdapter.extract` does no dtype cast.

* :class:`SharedExpertAdapter` -- replaces the inline shared/residual pathway of
  :class:`common.fusion_layer.FusionLayer`, whose ``forward`` computes
  ``shared = self.shared_expert(x)`` then
  ``shared = F.sigmoid(self.shared_expert_gate(x)) * shared``.
  :class:`Qwen15SharedExpert` reproduces that gated formula and additionally
  supports families with no shared expert (Mixtral / Qwen3): when the shared
  expert is absent it contributes nothing (``torch.zeros_like(x)``).

Adapters take explicit, named parameters so a typo is caught rather than
silently swallowed.

Library code only -- this module never imports an experiment ``config``.
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import nn

from common.mlp import AdvancedMLP
from common.mlp_utils import turn_in_mlp


def _require_float_weight(weight, what: str) -> torch.Tensor:
    """Validate that ``weight`` is a floating-point tensor or raise a clear error.

    Quantized experts expose ``None`` or an integer/packed ``weight`` instead of
    a float matrix; extracting them is unsupported, so fail with an explicit
    message rather than letting a downstream ``AttributeError`` / dtype error
    surface.
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
    """Extract a single MoE expert as an :class:`~common.mlp.AdvancedMLP`.

    The contract is :meth:`count` (how many experts a layer's container holds)
    and :meth:`extract` (build the ``AdvancedMLP`` for one expert on a given
    device, with no dtype cast -- the caller casts).
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
    """Expert extraction for the Qwen1.5-MoE / Mixtral-style and combined layouts.

    The layout is detected by whether ``experts`` is iterable (an
    ``nn.ModuleList`` of per-expert modules) or a single combined container
    (``for i, expert in enumerate(experts)`` raising ``TypeError``).
    """

    @staticmethod
    def _is_module_list(experts) -> bool:
        """True for the ModuleList layout (iterable), False for the combined one.

        Uses a ``try/except TypeError`` on ``enumerate(experts)``:
        ``enumerate`` calls ``iter(experts)``, so a non-iterable combined
        container raises ``TypeError`` and selects the combined branch.
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
        """ModuleList branch: delegate to ``turn_in_mlp`` (shares the projections).

        Delegates to ``turn_in_mlp(expert, device=device)``: the
        returned ``AdvancedMLP`` reuses ``expert.gate_proj`` / ``.up_proj`` /
        ``.down_proj`` directly (no clone). Validates the projection weights up
        front so a quantized expert fails with a clear error.
        """
        expert = experts[index]
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            proj = getattr(expert, proj_name, None)
            _require_float_weight(
                getattr(proj, "weight", None), f"expert {index} {proj_name}.weight"
            )
        return turn_in_mlp(expert, device=device)

    def _extract_combined(self, experts, index: int, device) -> AdvancedMLP:
        """Combined branch: build a fresh ``AdvancedMLP`` from sliced/cloned weights.

        The ``except TypeError`` branch: a single container
        exposes ``num_experts`` / ``hidden_dim`` / ``intermediate_dim`` and the
        stacked ``gate_up_proj`` (``[E, 2 * intermediate, hidden]``) and
        ``down_proj`` (``[E, hidden, intermediate]``) tensors. The gate/up halves
        are sliced out of ``gate_up_proj`` at ``intermediate_dim`` and every slice
        is detached and cloned (not shared).
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
# Shared-expert pathway adapter.
# --------------------------------------------------------------------------- #
class SharedExpertAdapter(ABC):
    """The shared/residual expert pathway as a callable module.

    The contract is :meth:`forward`, computing the family's shared-expert
    contribution for a hidden-state batch. Concrete adapters are
    ``nn.Module``-compatible callables so they move with ``.to(...)`` and slot
    into the layer like any submodule.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the shared-expert contribution for ``x``."""
        raise NotImplementedError()


class Qwen15SharedExpert(torch.nn.Module, SharedExpertAdapter):
    """Sigmoid-gated shared expert (Qwen1.5-MoE), with an absent-shared fallback.

    Reproduces the shared/residual pathway of
    :class:`common.fusion_layer.FusionLayer`:
    ``F.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)``.
    Families with no shared expert (Mixtral / Qwen3) pass ``None`` for both
    submodules and the pathway contributes zeros, so the residual add in the
    layer is a no-op. Presence/type is exposed through :attr:`has_shared` rather
    than any "exactly two ``.pt`` files" assumption.
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
