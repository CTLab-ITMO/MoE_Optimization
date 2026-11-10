"""Glue utilities and re-exports for the fusion pipeline.

Exposes ``FUSION_METHODS`` (name -> callable fusion instance, incl. ``LS-<name>``),
``best_metrics`` (name -> higher_is_better), and helpers for metric evaluation,
timing, and Qwen-module adaptation.
"""

import time
import warnings

import torch

from common.fusion import (
    AdvancedMLP,
    MLP,
    FUSION_REGISTRY,
    as_tensor,
)
from common.metrics import METRIC_REGISTRY, best_metrics
from common.names import TIME_METRIC
from common.numeric import _default_device

default_device = _default_device()

# Alias for FUSION_REGISTRY: name -> fusion instance.
# Callable as method(*mlps, train_data=...); stages read methods only from this registry.
FUSION_METHODS = FUSION_REGISTRY


@torch.no_grad()
def turn_in_mlp(mlp: torch.nn.Module, device=default_device) -> MLP:
    """Wrap a loaded Qwen/Mixtral-style MLP module as an :class:`AdvancedMLP`."""
    new_mlp = AdvancedMLP(
        mlp.gate_proj.in_features,
        mlp.gate_proj.out_features,
        mlp.down_proj.out_features,
        device=device,
    )
    new_mlp.gate = mlp.gate_proj
    new_mlp.up = mlp.up_proj
    new_mlp.down = mlp.down_proj
    return new_mlp


@torch.no_grad()
def timer(mlp: MLP, test_data, iters=3) -> float:
    """Apply ``mlp.forward_unzip`` on data and return the elapsed seconds."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    for _ in range(iters):
        mlp.forward_unzip(test_data)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    return end_time - start_time


@torch.no_grad()
def test_model_on_data(fused_mlp, mlp1, mlp2, test_data, ctx=None) -> dict[str, float]:
    """Compute all registered metrics plus wall-clock ``Time`` (ms) for a fused expert.

    ``ctx`` is passed to each metric. New metrics are included automatically.
    """
    all_tokens = as_tensor(test_data)
    real_res = torch.cat([mlp1.forward(all_tokens), mlp2.forward(all_tokens)], dim=-1).float()
    new_res = fused_mlp.forward_unzip(all_tokens).float()

    if not (torch.isfinite(new_res).all() and torch.isfinite(real_res).all()):
        warnings.warn(
            f"Non-finite output in fusion evaluation "
            f"(fused: {int(torch.isnan(new_res).sum())} NaN / "
            f"{int(torch.isinf(new_res).sum())} Inf; "
            f"reference non-finite: {not bool(torch.isfinite(real_res).all())}); "
            f"metrics set to NaN and excluded from averaging.",
            RuntimeWarning,
            stacklevel=2,
        )
        report = {name: float("nan") for name in METRIC_REGISTRY}
    else:
        report = {name: METRIC_REGISTRY[name](real_res, new_res, ctx=ctx) for name in METRIC_REGISTRY}
    report[TIME_METRIC] = 1000 * timer(fused_mlp, all_tokens)
    return report
