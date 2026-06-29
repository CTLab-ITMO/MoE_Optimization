"""Tests for the router-output canonicalizer (S7, §3.2).

Covers the S7 gate for the router adapter:

* **parity with the three inline call-sites** (the equivalence the golden gate
  protects): the bare-logits path reproduces stage 1's
  ``topk(softmax(logits, dim=1), k)`` and stage 5's ``topk(logits, k)``, and the
  tuple path reproduces stage 9's ``_, routing_weights, selected_experts =
  self.gate(...)``;
* the **monotonicity insight**: stage 1 (softmax then topk) and stage 5 (raw
  topk) select the *same* experts, so one unified adapter serves both;
* ``normalize_topk`` renormalizes the top-k weights to sum 1 only when set;
* the adapter works on both fake gates from ``common.testing.fakes``
  (``FakeRouterGate`` -> bare logits, ``FakeFusionGate`` -> 3-tuple);
* §3.7 housekeeping: the abstract base cannot be instantiated and the concrete
  ``route`` rejects an unknown keyword.
"""

import pytest
import torch

from common.router import Qwen15RouterAdapter, RouterAdapter, RouterDecision
from common.testing.fakes import H, TOP_K, FakeFusionGate, FakeRouterGate

SEED = 1234
T = 16
E = 8


def _logits(seed=SEED, t=T, e=E):
    torch.manual_seed(seed)
    return torch.randn(t, e)


# --------------------------------------------------------------------------- #
# Parity: bare-logits path == stage 1 / stage 5 inline.
# --------------------------------------------------------------------------- #
def test_bare_selected_matches_stage1_inline():
    logits = _logits()
    # Stage 1 inline (experts_fusion/1_generate_activation_matrix.py).
    routing_weights = logits.softmax(dim=1, dtype=torch.float)
    _, stage1_selected = torch.topk(routing_weights, TOP_K, dim=-1)

    decision = Qwen15RouterAdapter().route(logits, top_k=TOP_K)
    assert torch.equal(decision.selected_experts, stage1_selected)
    # Stage 1 uses softmax(dim=1); the adapter's softmax(dim=-1) is identical for
    # a 2D tensor, so the returned (top-k) weights match too.
    stage1_top_weights, _ = torch.topk(routing_weights, TOP_K, dim=-1)
    assert torch.equal(decision.routing_weights, stage1_top_weights)
    assert decision.routing_weights.dtype == torch.float


def test_bare_selected_matches_stage5_inline():
    logits = _logits()
    # Stage 5 inline (experts_fusion/5_activations.py): topk on RAW logits.
    _, stage5_selected = logits.topk(TOP_K, dim=-1)

    decision = Qwen15RouterAdapter().route(logits, top_k=TOP_K)
    assert torch.equal(decision.selected_experts, stage5_selected)


def test_stage1_and_stage5_select_identical_experts():
    """The monotonicity insight: softmax is order-preserving, so topk over the
    softmax (stage 1) and topk over the raw logits (stage 5) agree -- which is
    why a single adapter can stand in for both bare-logits call-sites."""
    logits = _logits()
    _, stage1 = torch.topk(logits.softmax(dim=1, dtype=torch.float), TOP_K, dim=-1)
    _, stage5 = logits.topk(TOP_K, dim=-1)
    assert torch.equal(stage1, stage5)


def test_bare_logits_field_is_the_input_tensor():
    logits = _logits()
    decision = Qwen15RouterAdapter().route(logits, top_k=TOP_K)
    assert decision.logits is logits


# --------------------------------------------------------------------------- #
# Parity: tuple path == stage 9 inline (FusionLayer).
# --------------------------------------------------------------------------- #
def test_tuple_path_matches_stage9_inline():
    torch.manual_seed(SEED)
    gate = FakeFusionGate()
    x = torch.randn(T, H)
    out = gate(x)  # (logits, routing_weights, selected_experts)

    # Stage 9 inline (common/fusion_layer.py): _, routing_weights, selected = gate(...)
    _, stage9_weights, stage9_selected = out

    decision = Qwen15RouterAdapter().route(out, top_k=TOP_K)
    assert torch.equal(decision.selected_experts, stage9_selected)
    assert torch.equal(decision.selected_experts, out[-1])
    assert torch.equal(decision.routing_weights, stage9_weights)
    assert torch.equal(decision.routing_weights, out[1])
    assert torch.equal(decision.logits, out[0])


# --------------------------------------------------------------------------- #
# normalize_topk (norm_topk_prob): off by default, renormalizes when on.
# --------------------------------------------------------------------------- #
def test_normalize_topk_renormalizes_to_sum_one():
    logits = _logits()
    decision = Qwen15RouterAdapter(normalize_topk=True).route(logits, top_k=TOP_K)
    sums = decision.routing_weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)


def test_default_does_not_renormalize():
    logits = _logits()
    decision = Qwen15RouterAdapter().route(logits, top_k=TOP_K)
    sums = decision.routing_weights.sum(dim=-1)
    # Top-k softmax mass is < 1 in general; assert it is genuinely not normalized.
    assert not torch.allclose(sums, torch.ones_like(sums), atol=1e-3)
    # And the unnormalized weights equal the raw top-k softmax slice.
    expected, _ = torch.topk(logits.softmax(dim=-1, dtype=torch.float), TOP_K, dim=-1)
    assert torch.equal(decision.routing_weights, expected)


def test_normalize_topk_preserves_selected_experts():
    """Renormalization scales the weights but must not change which experts win."""
    logits = _logits()
    plain = Qwen15RouterAdapter(normalize_topk=False).route(logits, top_k=TOP_K)
    normed = Qwen15RouterAdapter(normalize_topk=True).route(logits, top_k=TOP_K)
    assert torch.equal(plain.selected_experts, normed.selected_experts)


# --------------------------------------------------------------------------- #
# Works on both fake gates with the fakes' TOP_K.
# --------------------------------------------------------------------------- #
def test_route_on_fake_router_gate_bare_output():
    torch.manual_seed(SEED)
    gate = FakeRouterGate()
    x = torch.randn(T, H)
    out = gate(x)
    assert isinstance(out, torch.Tensor)  # bare logits

    decision = Qwen15RouterAdapter().route(out, top_k=TOP_K)
    _, expected = out.topk(TOP_K, dim=-1)
    assert torch.equal(decision.selected_experts, expected)
    assert decision.selected_experts.shape == (T, TOP_K)


def test_route_on_fake_fusion_gate_tuple_output():
    torch.manual_seed(SEED)
    gate = FakeFusionGate()
    x = torch.randn(T, H)
    out = gate(x)
    assert isinstance(out, tuple)

    decision = Qwen15RouterAdapter().route(out, top_k=TOP_K)
    assert torch.equal(decision.selected_experts, out[-1])
    assert torch.equal(decision.routing_weights, out[1])
    assert torch.equal(decision.logits, out[0])


# --------------------------------------------------------------------------- #
# §3.7 housekeeping: abstract base, explicit-param concrete signature, errors.
# --------------------------------------------------------------------------- #
def test_router_adapter_is_abstract():
    with pytest.raises(TypeError):
        RouterAdapter()


def test_concrete_route_rejects_unknown_kwarg():
    with pytest.raises(TypeError):
        Qwen15RouterAdapter().route(_logits(), top_k=TOP_K, bogus=1)


def test_route_rejects_non_tensor_non_tuple():
    with pytest.raises(TypeError):
        Qwen15RouterAdapter().route([1, 2, 3], top_k=TOP_K)


def test_router_decision_optional_fields_default_none():
    decision = RouterDecision(selected_experts=torch.tensor([0, 1]))
    assert decision.routing_weights is None
    assert decision.logits is None
