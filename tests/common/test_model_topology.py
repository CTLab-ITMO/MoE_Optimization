"""Tests for MoE-layer discovery (S7, §3.2).

Pins the behaviour that replaces the blind ``range(N)`` + ``del extra layers``
of stages 1/4/5:

* on a model where every layer is MoE (the :class:`FakeCausalLM`, standing in for
  Qwen1.5-MoE where all 24 layers are MoE) discovery is behaviour-NEUTRAL --
  ``discover_moe_layers(model) == list(range(n_layers))``;
* ``selected`` intersects while preserving the caller's order;
* on a synthetic *hybrid* stack (one dense ``nn.Linear`` mlp among MoE layers) the
  dense layer is reported non-MoE and excluded -- the case the old code crashed on;
* probing a bare object never raises.

Parity: the discovered set equals the indices the inline snippet would have
processed -- ``[i for i in range(len(model.model.layers))
if hasattr(model.model.layers[i].mlp, "experts")
and hasattr(model.model.layers[i].mlp, "gate")]``.
"""

import types

import pytest
import torch
from torch import nn

from common.model_topology import discover_moe_layers, is_moe_layer
from common.testing.fakes import FakeCausalLM, FakeMoEMLP


def _inline_discover(model):
    """The faithful inline equivalent the adapter replaces (parity oracle)."""
    layers = model.model.layers
    out = []
    for i in range(len(layers)):
        mlp = layers[i].mlp
        if hasattr(mlp, "experts") and hasattr(mlp, "gate"):
            out.append(i)
    return out


# --------------------------------------------------------------------------- #
# All-MoE model (Qwen1.5-MoE / FakeCausalLM): behaviour-neutral.
# --------------------------------------------------------------------------- #
def test_is_moe_layer_true_for_every_fake_layer():
    model = FakeCausalLM()
    assert all(is_moe_layer(layer) for layer in model.model.layers)


def test_discover_returns_all_indices_on_all_moe_model():
    model = FakeCausalLM()
    n = len(model.model.layers)
    assert discover_moe_layers(model) == list(range(n))


def test_discover_matches_inline_oracle_on_all_moe_model():
    # Parity: same indices the inline range()+hasattr snippet would process.
    model = FakeCausalLM()
    assert discover_moe_layers(model) == _inline_discover(model)


# --------------------------------------------------------------------------- #
# selected: intersect, preserving caller order; skip non-MoE / out-of-range.
# --------------------------------------------------------------------------- #
def test_selected_subset_preserved():
    model = FakeCausalLM()
    assert discover_moe_layers(model, selected=[0, 1]) == [0, 1]


def test_selected_order_is_preserved():
    model = FakeCausalLM()
    assert discover_moe_layers(model, selected=[1, 0]) == [1, 0]


def test_selected_skips_out_of_range_index():
    model = FakeCausalLM()
    n = len(model.model.layers)
    # n is out of range -> silently skipped, valid indices kept in given order.
    assert discover_moe_layers(model, selected=[n, 0]) == [0]


# --------------------------------------------------------------------------- #
# Synthetic hybrid dense+sparse stack: the case the old code crashed on.
# --------------------------------------------------------------------------- #
def _hybrid_model():
    """`.model.layers` with a dense (nn.Linear) mlp wedged between MoE layers."""
    moe0 = types.SimpleNamespace(mlp=FakeMoEMLP())
    dense = types.SimpleNamespace(mlp=nn.Linear(4, 4))  # no experts, no router gate
    moe2 = types.SimpleNamespace(mlp=FakeMoEMLP())
    inner = types.SimpleNamespace(layers=[moe0, dense, moe2])
    return types.SimpleNamespace(model=inner), dense


def test_dense_layer_is_not_moe():
    _, dense = _hybrid_model()
    assert is_moe_layer(dense) is False


def test_discover_excludes_dense_layer():
    model, _ = _hybrid_model()
    assert discover_moe_layers(model) == [0, 2]


def test_discover_hybrid_matches_inline_oracle():
    model, _ = _hybrid_model()
    assert discover_moe_layers(model) == _inline_discover(model)


def test_selected_among_hybrid_skips_dense_and_preserves_order():
    model, _ = _hybrid_model()
    # 1 is dense -> skipped; 2 and 0 kept in the requested order.
    assert discover_moe_layers(model, selected=[2, 1, 0]) == [2, 0]


# --------------------------------------------------------------------------- #
# Defensive: bare / malformed inputs never raise from is_moe_layer.
# --------------------------------------------------------------------------- #
def test_bare_object_without_mlp_is_not_moe():
    assert is_moe_layer(object()) is False


def test_layer_with_dense_mlp_object_is_not_moe():
    # A plain object with an mlp that lacks experts/gate is not MoE.
    layer = types.SimpleNamespace(mlp=types.SimpleNamespace(gate_proj=nn.Linear(4, 4)))
    assert is_moe_layer(layer) is False


def test_layer_with_only_gate_is_not_moe():
    # Router gate present but no experts collection -> not a MoE block.
    layer = types.SimpleNamespace(mlp=types.SimpleNamespace(gate=nn.Linear(4, 4)))
    assert is_moe_layer(layer) is False


# --------------------------------------------------------------------------- #
# Contract: a model without model.model.layers raises a clear error.
# --------------------------------------------------------------------------- #
def test_model_without_decoder_stack_raises():
    with pytest.raises(AttributeError):
        discover_moe_layers(types.SimpleNamespace())


def test_model_with_none_inner_raises():
    with pytest.raises(AttributeError):
        discover_moe_layers(types.SimpleNamespace(model=None))
