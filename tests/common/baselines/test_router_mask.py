import os

import pytest
import torch
from torch import nn

from common.artifacts import Group, Partition
from common.expert_adapter import Qwen15SharedExpert
from common.fused_group import FusedExpert, IdentityFusedGroup
from common.mlp import AdvancedMLP
from common.router import Qwen15RouterAdapter
from common import layout
from common.testing.builders import Builders, expert_mlp
from common.testing.fakes import N_EXPERTS as FAKE_N_EXPERTS
from common.fusion_layer import get_fusion_layer_from_groups, FusionLayer
from common.baselines.reap import MaskedRouterAdapter

CPU = torch.device("cpu")
HIDDEN = 4
NUM_EXPERTS = 4
TOP_K = 2


@pytest.fixture
def data_dir(tmp_path):
    return str(tmp_path)


class FakeGate(nn.Module):
    def __init__(self, num_experts=NUM_EXPERTS, top_k=TOP_K, hidden=HIDDEN):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.proj = nn.Linear(hidden, num_experts, bias=False)

    def forward(self, x):
        logits = self.proj(x)
        weights, selected = torch.topk(logits.softmax(dim=-1), self.top_k, dim=-1)
        return logits, weights, selected


class _ConstGate(nn.Module):
    """Gate returning fixed logits, as bare tensor or the 3-tuple form.

    Fixed logits let a test point the *unmasked* top-k at dropped experts, so it
    can assert the mask reselects survivors. The tuple form additionally computes
    ready weights/selected on the unmasked logits -- exactly what the mask must
    discard and recompute.
    """

    def __init__(self, logits, *, as_tuple, top_k):
        super().__init__()
        self.register_buffer("logits", logits)
        self.as_tuple = as_tuple
        self.top_k = top_k

    def forward(self, x):
        logits = self.logits.unsqueeze(0).expand(x.shape[0], -1)
        if not self.as_tuple:
            return logits
        weights, selected = torch.topk(logits.softmax(dim=-1), self.top_k, dim=-1)
        return logits, weights, selected


def _build_pruned_layer(gate, survivors, n_experts=NUM_EXPERTS, top_k=TOP_K):
    """Build a REAP-style layer: survivors are singleton (identity) groups, the
    rest are pruned from the router via a MaskedRouterAdapter."""
    torch.manual_seed(0)
    groups = [(s,) for s in survivors]
    fusion_experts = [AdvancedMLP(HIDDEN, 6, HIDDEN, device=CPU) for _ in groups]
    fusion_groups = [IdentityFusedGroup(mlp, 1) for mlp in fusion_experts]
    shared = Qwen15SharedExpert(
        nn.Linear(HIDDEN, HIDDEN, bias=False), nn.Linear(HIDDEN, 1, bias=False)
    )
    uncovered = [e for e in range(n_experts) if e not in survivors]
    layer = FusionLayer(
        experts=[None] * n_experts,
        expert_groups=groups,
        fusion_experts=fusion_experts,
        gate=gate,
        shared=shared,
        use_default_experts=False,
        fusion_groups=fusion_groups,
        top_k=top_k,
        router=MaskedRouterAdapter.from_uncovered(
            Qwen15RouterAdapter(), uncovered, n_experts, top_k
        ),
    )
    return layer, fusion_experts, shared


@pytest.mark.parametrize("as_tuple", [True, False])
def test_router_mask_reselects_survivor_over_dropped(as_tuple):
    """Both gate branches: a token whose unmasked top-k pointed at dropped experts
    reselects survivors, weighted by softmax renormalized over the survivors."""
    adapter = MaskedRouterAdapter.from_uncovered(
        Qwen15RouterAdapter(), uncovered=[3, 4], num_experts=5, top_k=2
    )
    logits = torch.tensor([0.1, 0.2, 0.3, 5.0, 4.0])
    gate = _ConstGate(logits, as_tuple=as_tuple, top_k=TOP_K)
    flat = torch.randn(2 * 3, HIDDEN)
    decision = adapter.route(gate(flat), top_k=2)
    selected = decision.selected_experts
    assert set(selected.flatten().tolist()) <= {0, 1, 2}
    assert 3 not in selected and 4 not in selected
    assert selected[0].tolist() == [2, 1]
    survivor_softmax = torch.tensor([0.1, 0.2, 0.3]).softmax(dim=-1)
    assert torch.allclose(decision.routing_weights[0], survivor_softmax[[2, 1]], atol=1e-6)


def test_router_mask_forward_matches_manual_masked_routing():
    """The full forward (reconstruct + mix + shared) equals a hand-rolled masked
    routing pass, for a random tuple gate with exactly top_k survivors."""
    torch.manual_seed(3)
    survivors = [0, 1]  # experts 2, 3 pruned; survivors == top_k
    gate = FakeGate()
    layer, fusion_experts, shared = _build_pruned_layer(gate, survivors, n_experts=NUM_EXPERTS)

    hidden = torch.randn(2, 3, HIDDEN)
    flat = hidden.view(-1, HIDDEN)
    mask = torch.zeros(NUM_EXPERTS)
    mask[[2, 3]] = float("-inf")
    with torch.no_grad():
        logits = gate(flat)[0]
        weights_full = (logits + mask).softmax(dim=-1, dtype=torch.float)
        w, sel = torch.topk(weights_full, TOP_K, dim=-1)
        expected = shared(flat).clone()
        for t in range(flat.shape[0]):
            for j in range(TOP_K):
                member = survivors.index(int(sel[t, j]))
                expected[t] += w[t, j] * fusion_experts[member](flat[t : t + 1])[0]
        out = layer.forward(hidden)
    assert set(sel.flatten().tolist()) <= set(survivors)
    assert torch.allclose(out, expected.view_as(hidden), atol=1e-6)


def test_router_mask_none_leaves_routing_untouched(data_dir):
    """Full coverage -> router_adapter left unchanged (no MaskedRouterAdapter),
    so routing stays byte-identical and forward is finite."""
    builders = Builders(data_dir, [0], FAKE_N_EXPERTS)
    builders.experts()
    pairs = builders.partition()  # covers every expert
    builders.fused_experts_v1("v1", pairs)
    layer = get_fusion_layer_from_groups(
        0, "v1", data_dir, device=CPU, use_default_experts=False,
        num_experts=FAKE_N_EXPERTS, router_adapter=Qwen15RouterAdapter(), top_k=TOP_K,
    )
    assert not isinstance(layer.router, MaskedRouterAdapter)
    out = layer.forward(torch.randn(1, 3, HIDDEN))
    assert torch.isfinite(out).all()


def test_masked_router_rejects_too_few_survivors():
    with pytest.raises(ValueError, match="fewer than"):
        MaskedRouterAdapter.from_uncovered(
            Qwen15RouterAdapter(), uncovered=[1, 2, 3], num_experts=4, top_k=2
        )


def test_masked_router_rejects_empty_uncovered():
    with pytest.raises(ValueError, match="at least one uncovered"):
        MaskedRouterAdapter.from_uncovered(
            Qwen15RouterAdapter(), uncovered=[], num_experts=4, top_k=2
        )


def test_loader_derives_masked_router_from_pruned_partition(data_dir):
    """A singleton (REAP) partition covering only some experts makes the loader
    wrap the router in a MaskedRouterAdapter that masks the pruned experts."""
    builders = Builders(data_dir, [0], FAKE_N_EXPERTS)
    builders.experts()
    survivors = [0, 1]  # experts 2, 3 pruned
    group_path = layout.group_path(data_dir, 0)
    os.makedirs(os.path.dirname(group_path), exist_ok=True)
    Partition(
        groups=tuple(Group(members=(s,)) for s in survivors), method="reap_saliency"
    ).save(group_path)
    for i in range(len(survivors)):
        fused_path = layout.fusion_expert_path(data_dir, "reap", 0, i)
        os.makedirs(os.path.dirname(fused_path), exist_ok=True)
        FusedExpert(expert_mlp(i), group_size=1, reconstruction="identity").save(fused_path)
    layer = get_fusion_layer_from_groups(
        0, "reap", data_dir, device=CPU, use_default_experts=False,
        num_experts=FAKE_N_EXPERTS, router_adapter=Qwen15RouterAdapter(), top_k=TOP_K,
    )
    assert isinstance(layer.router, MaskedRouterAdapter)
    assert torch.isneginf(layer.router.logit_mask[2]) and torch.isneginf(layer.router.logit_mask[3])
    assert layer.router.logit_mask[0].item() == 0.0 and layer.router.logit_mask[1].item() == 0.0
    out = layer.forward(torch.randn(1, 3, HIDDEN))
    assert out.shape == (1, 3, HIDDEN) and torch.isfinite(out).all()


def test_loader_full_coverage_partition_leaves_router_unmasked(data_dir):
    """A full-coverage partition leaves router_adapter unmasked."""
    builders = Builders(data_dir, [0], FAKE_N_EXPERTS)
    builders.experts()
    pairs = builders.partition()  # covers every expert
    builders.fused_experts_v1("v1", pairs)
    layer = get_fusion_layer_from_groups(
        0, "v1", data_dir, device=CPU, use_default_experts=False,
        num_experts=FAKE_N_EXPERTS, router_adapter=Qwen15RouterAdapter(), top_k=TOP_K,
    )
    assert not isinstance(layer.router, MaskedRouterAdapter)
