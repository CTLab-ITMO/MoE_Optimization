"""Container of dynamic per-stage state passed to strategies.

Strategies (fusion methods, grouping, metrics) receive runtime state through one
:class:`StageContext` instead of ad-hoc kwargs; static config stays in ``__init__``.
The dataclass is frozen; ``for_layer``/``for_group`` return cheap copies.
:meth:`generator` produces an order-independent :class:`torch.Generator` so
stochastic ops don't consume the global RNG stream.

Library code only -- never imports an experiment ``config``.
"""

from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional

import torch

from common.seeding import generator_for
from common.seeding import seed_for as _seed_for


@dataclass(frozen=True)
class StageContext:
    """Dynamic state shared with grouping, fusion, and metric strategies.

    All fields are optional. Key fields:

    * ``seed`` -- base seed; enables order-independent local seeds via :meth:`generator`.
    * ``layer_index`` / ``group_index`` -- current work-item coordinates, folded into derived seeds.
    * ``model`` / ``layer`` / ``gate`` / ``tokenizer`` -- live model handles.
    * ``n_experts`` -- expert count; grouping partitions this many experts.
    * ``device`` / ``dtype`` / ``data_dir`` -- tensor policy and artifact root.
    * ``calibration_data`` / ``validation_data`` / ``labels`` -- data handles (tensors or callables).
    * ``artifact_store`` / ``logger`` / ``run_metadata`` / ``config_metadata`` -- I/O and provenance.
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
        """Derive a seed for (stage, layer, group, **coords), or ``None`` if no base seed is set (caller uses global RNG)."""
        if self.seed is None:
            return None
        return _seed_for(seed=self.seed, stage=stage, **self._coords(coords))

    def generator(
        self, stage: str, *, device=None, **coords
    ) -> Optional[torch.Generator]:
        """Return an order-independent Generator for this work item, or ``None`` if no seed is set.

        ``device`` defaults to the context device then CPU; must match the device the sampler draws on.
        """
        if self.seed is None:
            return None
        dev = device if device is not None else self.device
        return generator_for(
            seed=self.seed, stage=stage, device=dev, **self._coords(coords)
        )
