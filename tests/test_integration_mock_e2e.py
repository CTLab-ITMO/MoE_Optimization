from pathlib import Path
import torch
import torch.nn as nn

from moe_opt_core.pipelines.h1_merge import run_h1_with_experts
from moe_opt_core.pipelines.h2_bias import denoise_bias_covariance
from moe_opt_core.pipelines.h3_routing import dynamic_topk_cumprob


class MockExpert(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.lin = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


def test_mock_e2e_small() -> None:
    d = 8
    experts = [MockExpert(d) for _ in range(4)]
    x = torch.randn(32, d)

    h1 = run_h1_with_experts(
        experts=experts,
        mlp_inputs=x,
        mode="pair",
        proj_dim=4,
        batch_size=8,
        seed=123,
        mp_factor=1.1,
        cluster_count=0,
    )
    assert len(h1.pairs) > 0

    imp = torch.randn(64, 4)
    h2 = denoise_bias_covariance(imp, alpha=0.2, lam=1e-3, normalize="maxabs")
    assert h2.bias.shape == (4,)

    probs = torch.softmax(torch.randn(16, 4), dim=-1)
    k = dynamic_topk_cumprob(probs, tau=0.9, k_min=1, k_max=4)
    assert int(k.min().item()) >= 1
