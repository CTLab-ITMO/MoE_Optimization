import torch

from moe_opt_core.pipelines.h3_routing import dynamic_topk_cumprob, routing_error_budget


def test_dynamic_topk_cumprob_bounds() -> None:
    probs = torch.tensor([[0.8, 0.1, 0.1], [0.34, 0.33, 0.33]], dtype=torch.float32)
    k = dynamic_topk_cumprob(probs, tau=0.9, k_min=1, k_max=3)
    assert k.shape == (2,)
    assert int(k.min().item()) >= 1
    assert int(k.max().item()) <= 3


def test_routing_error_budget_in_range() -> None:
    probs = torch.tensor([[0.7, 0.2, 0.1]], dtype=torch.float32)
    k = torch.tensor([2], dtype=torch.long)
    delta = routing_error_budget(probs, k)
    assert 0.0 <= float(delta.item()) <= 1.0
