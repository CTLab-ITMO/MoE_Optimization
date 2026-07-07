"""Tests for the expert + shared-expert adapters in ``common.expert_adapter``.

These adapters are standalone (not yet wired into stages), so every test builds
its own inputs from ``common.testing.fakes`` and asserts the adapter reproduces
the inline logic it replaces:

* :class:`Qwen15ExpertAdapter` vs stage 3 (``experts_fusion/3_save_experts.py``)
  -- both the ModuleList branch (``turn_in_mlp``, shared projections) and the
  combined branch (sliced/cloned ``gate_up_proj`` / ``down_proj``);
* :class:`Qwen15SharedExpert` vs the sigmoid-gated shared pathway of
  :class:`common.fusion_layer.FusionLayer`.
"""

import types

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from common.expert_adapter import (
    ExpertAdapter,
    MixtralExpertAdapter,
    NoSharedExpertAdapter,
    Qwen15ExpertAdapter,
    Qwen15SharedExpert,
    Qwen15SharedExpertAdapter,
    SharedExpertAdapter,
)
from common.mlp import AdvancedMLP
from common.mlp_utils import turn_in_mlp
from common.testing.fakes import (
    H,
    INTER,
    N_EXPERTS,
    FakeExpert,
    FakeMixtralExpert,
    FakeMoEMLP,
)

CPU = torch.device("cpu")


class _FakeCombinedExperts(nn.Module):
    """Synthetic combined-layout container (Qwen3-style stacked projections).

    Exposes ``num_experts`` / ``hidden_dim`` / ``intermediate_dim`` and the
    stacked ``gate_up_proj`` (``[E, 2*INTER, H]``) / ``down_proj``
    (``[E, H, INTER]``) -- and is deliberately non-iterable so the adapter
    selects the combined branch (as stage 3 does on ``TypeError``).
    """

    def __init__(self, num_experts: int, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.gate_up_proj = nn.Parameter(
            torch.randn(num_experts, 2 * intermediate_dim, hidden_dim)
        )
        self.down_proj = nn.Parameter(
            torch.randn(num_experts, hidden_dim, intermediate_dim)
        )


# --------------------------------------------------------------------------- #
# ModuleList branch.
# --------------------------------------------------------------------------- #
def test_count_module_list():
    experts = nn.ModuleList([FakeExpert() for _ in range(N_EXPERTS)])
    assert Qwen15ExpertAdapter().count(experts) == N_EXPERTS


def test_extract_module_list_matches_turn_in_mlp():
    torch.manual_seed(0)
    experts = nn.ModuleList([FakeExpert() for _ in range(3)])
    adapter = Qwen15ExpertAdapter()
    x = torch.randn(5, H)
    for i in range(len(experts)):
        got = adapter.extract(experts, i, device=CPU)
        ref = turn_in_mlp(experts[i], device=CPU)
        assert isinstance(got, AdvancedMLP)
        # ModuleList branch SHARES the projection modules (no clone).
        assert got.gate is experts[i].gate_proj
        assert got.up is experts[i].up_proj
        assert got.down is experts[i].down_proj
        assert torch.equal(got.forward(x), ref.forward(x))


# --------------------------------------------------------------------------- #
# Combined branch.
# --------------------------------------------------------------------------- #
def test_count_combined_uses_num_experts():
    experts = _FakeCombinedExperts(num_experts=N_EXPERTS, hidden_dim=H, intermediate_dim=INTER)
    assert Qwen15ExpertAdapter().count(experts) == N_EXPERTS


def test_extract_combined_matches_inline_slicing():
    torch.manual_seed(0)
    n_experts, hidden, inter = 3, H, INTER
    experts = _FakeCombinedExperts(n_experts, hidden, inter)
    adapter = Qwen15ExpertAdapter()
    x = torch.randn(5, hidden)
    for i in range(n_experts):
        got = adapter.extract(experts, i, device=CPU)
        # Hand-built inline reference (the ``except TypeError`` branch of stage 3).
        ref = AdvancedMLP(hidden, inter, hidden, device=CPU)
        ref.gate.weight = nn.Parameter(experts.gate_up_proj[i, :inter].detach().clone())
        ref.up.weight = nn.Parameter(experts.gate_up_proj[i, inter:].detach().clone())
        ref.down.weight = nn.Parameter(experts.down_proj[i].detach().clone())

        assert isinstance(got, AdvancedMLP)
        assert torch.equal(got.gate.weight, ref.gate.weight)
        assert torch.equal(got.up.weight, ref.up.weight)
        assert torch.equal(got.down.weight, ref.down.weight)
        assert torch.equal(got.forward(x), ref.forward(x))


def test_extract_combined_clones_weights():
    """Combined branch CLONES: mutating the source must not touch the extract."""
    experts = _FakeCombinedExperts(num_experts=2, hidden_dim=H, intermediate_dim=INTER)
    got = Qwen15ExpertAdapter().extract(experts, 0, device=CPU)
    before = got.gate.weight.detach().clone()
    with torch.no_grad():
        experts.gate_up_proj.zero_()
    assert torch.equal(got.gate.weight, before)


# --------------------------------------------------------------------------- #
# MixtralExpertAdapter: reference second family (w1/w3/w2 ModuleList).
# --------------------------------------------------------------------------- #
def test_mixtral_count_module_list():
    experts = nn.ModuleList([FakeMixtralExpert() for _ in range(N_EXPERTS)])
    assert MixtralExpertAdapter().count(experts) == N_EXPERTS


def test_mixtral_extract_shares_w1_w3_w2_as_gate_up_down():
    """Mixtral extract wraps w1->gate, w3->up, w2->down, sharing the Linears."""
    torch.manual_seed(0)
    experts = nn.ModuleList([FakeMixtralExpert() for _ in range(3)])
    adapter = MixtralExpertAdapter()
    for i in range(len(experts)):
        got = adapter.extract(experts, i, device=CPU)
        assert isinstance(got, AdvancedMLP)
        # ModuleList branch SHARES the projection modules (no clone).
        assert got.gate is experts[i].w1
        assert got.up is experts[i].w3
        assert got.down is experts[i].w2


def test_mixtral_extract_forward_matches_expert_forward():
    """The wrapped AdvancedMLP reproduces the Mixtral expert's own forward."""
    torch.manual_seed(0)
    experts = nn.ModuleList([FakeMixtralExpert() for _ in range(3)])
    adapter = MixtralExpertAdapter()
    x = torch.randn(5, H)
    for i in range(len(experts)):
        got = adapter.extract(experts, i, device=CPU)
        assert torch.equal(got.forward(x), experts[i].forward(x))


def test_mixtral_is_expert_adapter_subclass():
    assert isinstance(MixtralExpertAdapter(), ExpertAdapter)


def test_mixtral_quantized_none_weight_raises():
    experts = nn.ModuleList([FakeMixtralExpert()])
    experts[0].w1.weight = None  # simulate a quantized projection
    with pytest.raises(TypeError, match="quantized experts are not supported"):
        MixtralExpertAdapter().extract(experts, 0, device=CPU)


# --------------------------------------------------------------------------- #
# Quantized experts -> clear error.
# --------------------------------------------------------------------------- #
def test_quantized_module_list_none_weight_raises():
    experts = nn.ModuleList([FakeExpert()])
    experts[0].gate_proj.weight = None  # simulate a quantized projection
    with pytest.raises(TypeError, match="quantized experts are not supported"):
        Qwen15ExpertAdapter().extract(experts, 0, device=CPU)


def test_quantized_combined_int_weight_raises():
    # A combined container whose ``gate_up_proj`` is an integer (packed) tensor.
    experts = types.SimpleNamespace(
        num_experts=2,
        hidden_dim=H,
        intermediate_dim=INTER,
        gate_up_proj=torch.zeros(2, 2 * INTER, H, dtype=torch.long),
        down_proj=torch.zeros(2, H, INTER),
    )
    with pytest.raises(TypeError, match="quantized experts are not supported"):
        Qwen15ExpertAdapter().extract(experts, 0, device=CPU)


# --------------------------------------------------------------------------- #
# Shared-expert runtime module (Qwen15SharedExpert).
# --------------------------------------------------------------------------- #
def test_shared_expert_matches_inline_formula():
    torch.manual_seed(0)
    moe = FakeMoEMLP()
    runtime = Qwen15SharedExpert(moe.shared_expert, moe.shared_expert_gate)
    assert isinstance(runtime, torch.nn.Module)
    assert runtime.has_shared is True

    x = torch.randn(7, H)
    got = runtime(x)  # exercise the nn.Module __call__ path
    # Inline FusionLayer pathway: F.sigmoid(gate(x)) * shared(x).
    ref = F.sigmoid(moe.shared_expert_gate(x)) * moe.shared_expert(x)
    assert torch.equal(got, ref)


def test_shared_expert_submodules_registered():
    moe = FakeMoEMLP()
    runtime = Qwen15SharedExpert(moe.shared_expert, moe.shared_expert_gate)
    children = dict(runtime.named_children())
    assert children["shared_expert"] is moe.shared_expert
    assert children["shared_expert_gate"] is moe.shared_expert_gate


def test_shared_expert_none_returns_zeros():
    runtime = Qwen15SharedExpert(None, None)
    assert runtime.has_shared is False
    x = torch.randn(7, H)
    out = runtime(x)
    assert torch.equal(out, torch.zeros_like(x))
    assert out.shape == x.shape


# --------------------------------------------------------------------------- #
# Shared-expert DI adapter (extract + build).
# --------------------------------------------------------------------------- #
def test_qwen15_shared_adapter_extract_returns_submodules():
    moe = FakeMoEMLP()
    adapter = Qwen15SharedExpertAdapter()
    assert isinstance(adapter, SharedExpertAdapter)
    extracted = adapter.extract(moe)
    assert extracted == (moe.shared_expert, moe.shared_expert_gate)


def test_qwen15_shared_adapter_extract_none_when_no_shared():
    """A module without a ``shared_expert`` attribute degrades to None, not an error."""
    bare = types.SimpleNamespace()  # no shared_expert / shared_expert_gate
    assert Qwen15SharedExpertAdapter().extract(bare) is None


def test_qwen15_shared_adapter_build_roundtrips_formula():
    torch.manual_seed(0)
    moe = FakeMoEMLP()
    adapter = Qwen15SharedExpertAdapter()
    runtime = adapter.build(adapter.extract(moe))
    assert isinstance(runtime, Qwen15SharedExpert)
    assert runtime.has_shared is True

    x = torch.randn(5, H)
    ref = F.sigmoid(moe.shared_expert_gate(x)) * moe.shared_expert(x)
    assert torch.equal(runtime(x), ref)


def test_qwen15_shared_adapter_build_none_is_zeros():
    runtime = Qwen15SharedExpertAdapter().build(None)
    assert runtime.has_shared is False
    x = torch.randn(5, H)
    assert torch.equal(runtime(x), torch.zeros_like(x))


# --------------------------------------------------------------------------- #
# No-shared-expert fallback adapter (families like Mixtral).
# --------------------------------------------------------------------------- #
def test_no_shared_adapter_extract_is_always_none():
    """Even a module that HAS a shared expert yields None: the family declares none."""
    adapter = NoSharedExpertAdapter()
    assert isinstance(adapter, SharedExpertAdapter)
    assert adapter.extract(FakeMoEMLP()) is None


def test_no_shared_adapter_build_is_zeros():
    runtime = NoSharedExpertAdapter().build(None)
    assert runtime.has_shared is False
    x = torch.randn(5, H)
    out = runtime(x)
    assert torch.equal(out, torch.zeros_like(x))
    assert out.shape == x.shape


# --------------------------------------------------------------------------- #
# Abstract base contracts.
# --------------------------------------------------------------------------- #
def test_abstract_bases_cannot_instantiate():
    with pytest.raises(TypeError):
        ExpertAdapter()
    with pytest.raises(TypeError):
        SharedExpertAdapter()
