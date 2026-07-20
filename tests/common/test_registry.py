"""Tests for the in-tree registry and the frozen k-agnostic fusion contract (S5).

Covers the S5 gate:

* generic :class:`Registry` -- ordered Mapping, collision-checked registration
  (``override=False``), ``names()``, entry-point discovery;
* config validation driven from ``Registry.names()`` (``require_registered`` /
  ``require_all_registered``);
* the canonical fusion / metric / grouping registries are pinned to the same
  composition *and order* the golden ``registry`` snapshot guards;
* the frozen k-agnostic fusion signature: the legacy positional
  ``method(a, b, data)`` and the variadic ``method(*[a, b], train_data=data)``
  produce a numerically identical merge for every method; the per-class arity
  contract (``supported_arity``): pairwise methods reject ``k != 2``,
  k-agnostic methods (Average / Pruning) accept any ``k >= 1``;
* the least-squares adapter's public ``refit_unmerge`` reproduces the old
  in-line ``.down``/``.U`` solve bit-for-bit.
"""

import pytest
import torch

from common import mlp_utils
from common.fusion import (
    FUSION_REGISTRY,
    AverageFusion,
    BASE_FUSION_METHODS,
    LeastSquaresAdapter,
    PruningFusion,
    ZipItFusion,
    build_fusion_methods,
)
from common.metrics import METRIC_REGISTRY, METRICS
from common.mlp import AdvancedMLP
from common.names import PairingMethod
from common.grouping import GROUPING_REGISTRY
from common.numeric import solve_linear
from common.registry import (
    Registry,
    RegistryError,
    require_all_registered,
    require_registered,
)

CPU = torch.device("cpu")

EXPECTED_FUSION_ORDER = [
    "ZipIt", "Average", "Pruning", "Procrustes", "Fisher", "TIES", "Distillation",
    "FrequencyWeightedAverage",
    "LS_ZipIt", "LS_Average", "LS_Pruning", "LS_Procrustes", "LS_Fisher",
    "LS_TIES", "LS_Distillation",
]


def _pair(seed_a=1, seed_b=2, in_f=4, hid=6, out=4):
    torch.manual_seed(seed_a)
    a = AdvancedMLP(in_f, hid, out, device=CPU)
    torch.manual_seed(seed_b)
    b = AdvancedMLP(in_f, hid, out, device=CPU)
    return a, b


def _states_equal(m1, m2) -> bool:
    s1, s2 = m1.state_dict(), m2.state_dict()
    return set(s1) == set(s2) and all(torch.equal(s1[k], s2[k]) for k in s1)


# --------------------------------------------------------------------------- #
# Generic Registry behavior.
# --------------------------------------------------------------------------- #
def test_registry_is_ordered_mapping():
    reg = Registry("t")
    reg.register("a", 1)
    reg.register("b", 2)
    assert reg.names() == ["a", "b"]
    assert reg["a"] == 1
    assert "b" in reg
    assert len(reg) == 2
    assert list(reg.keys()) == ["a", "b"]
    assert list(reg.items()) == [("a", 1), ("b", 2)]
    assert list(reg.values()) == [1, 2]


def test_registry_register_as_decorator():
    reg = Registry("t")

    @reg.register("plugin")
    class Plugin:
        pass

    assert reg["plugin"] is Plugin


def test_registry_collision_raises_without_override():
    reg = Registry("t")
    reg.register("a", 1)
    with pytest.raises(RegistryError):
        reg.register("a", 2)
    assert reg["a"] == 1  # unchanged


def test_registry_override_replaces():
    reg = Registry("t")
    reg.register("a", 1)
    reg.register("a", 2, override=True)
    assert reg["a"] == 2


def test_registry_unknown_key_raises_registry_error():
    reg = Registry("t")
    with pytest.raises(RegistryError):
        reg["missing"]


def test_registry_rejects_non_str_key():
    reg = Registry("t")
    with pytest.raises(RegistryError):
        reg.register(3, "x")


# --------------------------------------------------------------------------- #
# Config validation from Registry.names().
# --------------------------------------------------------------------------- #
def test_require_registered_accepts_valid_and_none():
    require_registered(None, FUSION_REGISTRY, field="X")  # no raise
    require_registered("Average", FUSION_REGISTRY, field="X")  # no raise


def test_require_registered_rejects_unknown_with_options():
    with pytest.raises(ValueError) as excinfo:
        require_registered("Nope", FUSION_REGISTRY, field="INSPECT_FUSION_METHOD")
    message = str(excinfo.value)
    assert "INSPECT_FUSION_METHOD" in message
    assert "Average" in message  # the valid set is spelled out


def test_require_all_registered_handles_none_empty_and_unknown():
    require_all_registered(None, FUSION_REGISTRY, field="FILTER_FUSIONS")  # no raise
    require_all_registered([], FUSION_REGISTRY, field="FILTER_FUSIONS")  # no raise
    require_all_registered(["Average", "ZipIt"], FUSION_REGISTRY, field="FILTER_FUSIONS")
    with pytest.raises(ValueError):
        require_all_registered(["Average", "bogus"], FUSION_REGISTRY, field="FILTER_FUSIONS")


# --------------------------------------------------------------------------- #
# Canonical registries: pinned order (golden snapshot).
# --------------------------------------------------------------------------- #
def test_fusion_registry_order_is_pinned():
    assert FUSION_REGISTRY.names() == EXPECTED_FUSION_ORDER


def test_build_fusion_methods_matches_registry_order():
    assert list(build_fusion_methods().keys()) == FUSION_REGISTRY.names()


def test_mlp_utils_reads_from_registry():
    # The stages import FUSION_METHODS from mlp_utils; it must *be* the registry.
    assert mlp_utils.FUSION_METHODS is FUSION_REGISTRY


def test_metric_registry_order_matches_metrics():
    assert METRIC_REGISTRY.names() == [metric.name for metric in METRICS]


def test_grouping_registry_matches_pairing_methods():
    assert GROUPING_REGISTRY.names() == [str(method) for method in PairingMethod]


# --------------------------------------------------------------------------- #
# Frozen k-agnostic fusion contract.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", EXPECTED_FUSION_ORDER)
def test_legacy_positional_equals_k_agnostic_keyword(name):
    """method(a, b, data) (legacy) == method(*[a, b], train_data=data) (k-agnostic)."""
    method = FUSION_REGISTRY[name]
    a, b = _pair()
    torch.manual_seed(64)
    data = torch.randn(64, 4)

    torch.manual_seed(7)
    legacy = method(a, b, data)
    torch.manual_seed(7)
    k_agnostic = method(*[a, b], train_data=data)
    assert _states_equal(legacy, k_agnostic)


PAIRWISE_METHODS = [
    name for name in EXPECTED_FUSION_ORDER
    if FUSION_REGISTRY[name].supported_arity == 2
]
K_AGNOSTIC_METHODS = [
    name for name in EXPECTED_FUSION_ORDER
    if FUSION_REGISTRY[name].supported_arity is None
]


def test_arity_split_is_pinned():
    # Average / Pruning / FrequencyWeightedAverage (and nothing else) are
    # k-agnostic; every LS_* wrapper stays strictly pairwise even when it wraps a
    # k-agnostic base method.
    assert K_AGNOSTIC_METHODS == ["Average", "Pruning", "FrequencyWeightedAverage"]
    assert set(PAIRWISE_METHODS) | set(K_AGNOSTIC_METHODS) == set(EXPECTED_FUSION_ORDER)


@pytest.mark.parametrize("name", PAIRWISE_METHODS)
def test_pairwise_methods_reject_non_pair_arity(name):
    method = FUSION_REGISTRY[name]
    a, b = _pair()
    c, _ = _pair(seed_a=3)
    data = torch.randn(8, 4)
    with pytest.raises(NotImplementedError) as excinfo:
        method(a, b, c, train_data=data)
    assert "k=2" in str(excinfo.value)
    with pytest.raises(NotImplementedError):
        method(a, train_data=data)


@pytest.mark.parametrize("name", K_AGNOSTIC_METHODS)
def test_k_agnostic_methods_accept_k1_and_k3(name):
    method = FUSION_REGISTRY[name]
    a, b = _pair()
    c, _ = _pair(seed_a=3)
    data = torch.randn(8, 4)
    single = method(a, train_data=data)
    assert torch.equal(single.gate.weight, a.gate.weight)
    triple = method(a, b, c, train_data=data)
    assert triple.gate.weight.shape == a.gate.weight.shape


def test_call_rejects_zero_experts():
    with pytest.raises(ValueError):
        AverageFusion()(train_data=torch.randn(8, 4))


def test_call_third_positional_module_is_third_expert():
    # Parse rule for the legacy shim: a third positional nn.Module is a third
    # expert (train_data for k >= 3 is keyword-only), not train_data.
    a, b = _pair()
    c, _ = _pair(seed_a=3)
    merged = AverageFusion()(a, b, c)
    expected = (a.gate.weight + b.gate.weight + c.gate.weight) / 3
    assert torch.allclose(merged.gate.weight, expected)
    # For a pairwise method the same call is an arity error, not a silent bind.
    with pytest.raises(NotImplementedError):
        ZipItFusion()(a, b, c)


def test_call_legacy_three_positional_binds_train_data():
    a, b = _pair()
    data = torch.randn(8, 4)
    # Pruning is data-free: 3rd positional must be bound as train_data, not an
    # expert (which would raise the k>2 error).
    merged = PruningFusion()(a, b, data)
    assert torch.equal(merged.gate.weight, a.gate.weight)


def test_call_rejects_unknown_kwarg():
    # The interface is strict: an unknown keyword is a TypeError, not silently
    # swallowed (extension is by editing the signature in-tree, not forward-compat).
    a, b = _pair()
    data = torch.randn(8, 4)
    with pytest.raises(TypeError):
        AverageFusion()(a, b, train_data=data, future_option=123)


# --------------------------------------------------------------------------- #
# Least-squares adapter: public refit_unmerge == old in-line solve.
# --------------------------------------------------------------------------- #
def test_ls_refit_unmerge_matches_manual_solve():
    a, b = _pair()
    torch.manual_seed(0)
    train = torch.randn(64, 4)

    # Manual, old-style refit on an independent Average-fused copy.
    reference = AverageFusion()(a, b, train_data=train)
    dtype = reference.down.weight.dtype
    reference.down = None
    new_res = reference.forward(train)
    targets = torch.cat([a.forward(train), b.forward(train)], dim=-1)
    u_manual = solve_linear(new_res.float(), targets.float()).to(dtype=dtype)

    # Via the LS adapter, which now goes through AdvancedMLP.refit_unmerge.
    ls_fused = LeastSquaresAdapter(AverageFusion())(a, b, train_data=train)
    assert ls_fused.down is None
    assert ls_fused.U is not None
    assert torch.allclose(ls_fused.U, u_manual, atol=1e-6)


def test_base_fusion_methods_all_in_registry():
    for method in BASE_FUSION_METHODS:
        assert FUSION_REGISTRY[method.name] is method
        # Only wrappable base methods get an LS_<name> variant; wrappable=False
        # methods (e.g. FrequencyWeightedAverage) are intentionally left bare.
        if method.wrappable:
            assert FUSION_REGISTRY[f"LS_{method.name}"].name == f"LS_{method.name}"
        else:
            assert f"LS_{method.name}" not in FUSION_REGISTRY
