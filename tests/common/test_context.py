"""Tests for :class:`common.context.StageContext` (S4).

The context is the single carrier of dynamic per-stage state passed to extension
points (§3.7). These tests cover the parts S4 actually relies on: it is frozen,
its derived copies pin layer/group coordinates, and its seeding helpers
(:meth:`StageContext.seed_for` / :meth:`StageContext.generator`) fold those
coordinates into the order-independent ``seed_for`` and fall back to the global
RNG (return ``None``) when no base seed is set.
"""

import dataclasses

import pytest
import torch

from common.context import StageContext
from common.seeding import seed_for


def test_defaults_are_all_none():
    ctx = StageContext()
    assert ctx.seed is None
    assert ctx.layer_index is None
    assert ctx.group_index is None
    assert ctx.device is None


def test_is_frozen():
    ctx = StageContext(seed=1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.seed = 2  # type: ignore[misc]


def test_for_layer_and_for_group_return_new_pinned_copies():
    base = StageContext(seed=42, device=torch.device("cpu"))
    layer_ctx = base.for_layer(3)
    assert base.layer_index is None  # original untouched
    assert layer_ctx.layer_index == 3 and layer_ctx.group_index is None
    assert layer_ctx.seed == 42 and layer_ctx.device == torch.device("cpu")

    group_ctx = layer_ctx.for_group(2)
    assert group_ctx.layer_index == 3 and group_ctx.group_index == 2


def test_for_layer_clears_group():
    ctx = StageContext(seed=1).for_group(5).for_layer(1)
    assert ctx.group_index is None and ctx.layer_index == 1


def test_seed_for_folds_in_layer_and_group():
    ctx = StageContext(seed=42).for_layer(3).for_group(1)
    assert ctx.seed_for("fusion", method="Fisher") == seed_for(
        seed=42, stage="fusion", layer=3, group=1, method="Fisher"
    )


def test_explicit_coords_override_pinned_ones():
    ctx = StageContext(seed=42).for_layer(1)
    assert ctx.seed_for("fusion", method="Fisher", group=7) == seed_for(
        seed=42, stage="fusion", layer=1, group=7, method="Fisher"
    )


def test_seed_for_and_generator_are_none_without_base_seed():
    ctx = StageContext().for_layer(3)
    assert ctx.seed_for("fusion", method="Fisher") is None
    assert ctx.generator("fusion", method="Fisher") is None


def test_generator_is_reproducible_and_coord_sensitive():
    base = StageContext(seed=42)
    g1 = base.for_layer(1).for_group(0).generator("fusion", method="Fisher")
    g2 = base.for_layer(1).for_group(0).generator("fusion", method="Fisher")
    g3 = base.for_layer(2).for_group(0).generator("fusion", method="Fisher")
    a = torch.randn(4, generator=g1)
    b = torch.randn(4, generator=g2)
    c = torch.randn(4, generator=g3)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_generator_returns_torch_generator():
    ctx = StageContext(seed=1).for_layer(0)
    assert isinstance(ctx.generator("fusion", method="Fisher"), torch.Generator)


def test_optional_handle_fields_are_carried():
    sentinel_model = object()
    ctx = StageContext(seed=1, model=sentinel_model, dtype=torch.float32)
    assert ctx.model is sentinel_model
    assert ctx.dtype == torch.float32
    # Derived copies preserve the handles.
    assert ctx.for_layer(0).model is sentinel_model


def test_grouping_state_fields_default_none_and_are_carried():
    # The grouping plugin reads the run's artifact root + expert count from ctx
    # (S6): both default to None and survive the per-layer derived copy.
    assert StageContext().data_dir is None
    assert StageContext().n_experts is None
    ctx = StageContext(seed=1, data_dir="/tmp/run", n_experts=60).for_layer(2)
    assert ctx.data_dir == "/tmp/run"
    assert ctx.n_experts == 60
