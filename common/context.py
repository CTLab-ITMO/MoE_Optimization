"""Single object of dynamic per-stage state passed to extension points (§3.7).

A plugin (fusion method, grouping strategy, metric) receives the *dynamic* state
it might need through one :class:`StageContext`, rather than through a bag of
``**kwargs`` the driver fills in blindly (§2.5). Static configuration of a method
stays in its ``__init__``; everything that varies per run / per layer / per group
travels in the context. This is what makes the k-agnostic signature freeze (S5)
possible: a method's signature can be ``(*mlps, ctx, **kwargs)`` because all the
moving parts live behind ``ctx``.

Every field is optional and the dataclass is frozen: contexts are cheap, derived
values (``for_layer`` / ``for_group``) return new instances, and new fields may
only be added as further optional fields (additive extension, §1.8/§2.5).

Its main job in S4 is to carry the *seeding* coordinates: :meth:`generator`
returns an order-independent :class:`torch.Generator` (via
:func:`common.seeding.seed_for`) for the layer/group the context points at, so a
stochastic op no longer consumes the global RNG stream.

Library code only -- this module never imports an experiment ``config``.
"""

from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional

import torch

from common.seeding import generator_for
from common.seeding import seed_for as _seed_for


@dataclass(frozen=True)
class StageContext:
    """Dynamic per-stage state shared with grouping / fusion / metric plugins.

    Fields (all optional):

    * ``seed`` -- run-level base seed; with it set, :meth:`generator` /
      :meth:`seed_for` derive order-independent local seeds, otherwise they
      return ``None`` and callers fall back to the global RNG.
    * ``layer_index`` / ``group_index`` -- coordinates of the current work item,
      folded into the derived seed automatically.
    * ``model`` / ``layer`` / ``gate`` / ``tokenizer`` -- handles onto the live
      model the plugin may need (router adapter, expert weights, ...).
    * ``n_experts`` -- number of experts in the current layer (model-derived);
      a grouping strategy partitions this many experts.
    * ``device`` / ``dtype`` -- device + dtype policy for newly created tensors.
    * ``data_dir`` -- the run's artifact root; readers of intermediate artifacts
      (e.g. a grouping :class:`~common.grouping.AffinityProvider` loading the
      co-activation matrix) resolve their paths under it via :mod:`common.layout`.
    * ``calibration_data`` / ``validation_data`` / ``labels`` -- lazily provided
      data handles (a tensor, a list or a zero-arg callable returning one).
    * ``artifact_store`` / ``logger`` -- I/O and logging sinks.
    * ``run_metadata`` / ``config_metadata`` -- provenance maps for the run.
    """

    seed: Optional[int] = None
    layer_index: Optional[int] = None
    group_index: Optional[int] = None
    model: Optional[Any] = None
    layer: Optional[Any] = None
    gate: Optional[Any] = None
    tokenizer: Optional[Any] = None
    n_experts: Optional[int] = None
    device: Optional[torch.device] = None
    dtype: Optional[torch.dtype] = None
    data_dir: Optional[str] = None
    calibration_data: Optional[Any] = None
    validation_data: Optional[Any] = None
    labels: Optional[Any] = None
    artifact_store: Optional[Any] = None
    logger: Optional[Any] = None
    run_metadata: Optional[Mapping[str, Any]] = None
    config_metadata: Optional[Mapping[str, Any]] = None

    def for_layer(self, layer_index: Optional[int]) -> "StageContext":
        """Return a copy pinned to ``layer_index`` (and cleared group)."""
        return replace(self, layer_index=layer_index, group_index=None)

    def for_group(self, group_index: Optional[int]) -> "StageContext":
        """Return a copy pinned to ``group_index`` within the current layer."""
        return replace(self, group_index=group_index)

    def _coords(self, coords: dict) -> dict:
        """Merge the context's own layer/group coordinates with explicit ones."""
        merged = {"layer": self.layer_index, "group": self.group_index}
        merged.update(coords)
        return merged

    def seed_for(self, stage: str, **coords) -> Optional[int]:
        """Derived seed for ``(stage, layer, group, **coords)`` or ``None``.

        Returns ``None`` when no base ``seed`` is set, signalling callers to keep
        using the global RNG (backward-compatible default).
        """
        if self.seed is None:
            return None
        return _seed_for(seed=self.seed, stage=stage, **self._coords(coords))

    def generator(
        self, stage: str, *, device=None, **coords
    ) -> Optional[torch.Generator]:
        """Order-independent :class:`torch.Generator` for this work item.

        ``None`` when no base ``seed`` is set (caller falls back to global RNG).
        ``device`` defaults to the context device, then CPU; it must match the
        device the sampler draws on.
        """
        if self.seed is None:
            return None
        dev = device if device is not None else self.device
        return generator_for(
            seed=self.seed, stage=stage, device=dev, **self._coords(coords)
        )
