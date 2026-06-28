"""Registries and glue utilities tying together fusion methods and metrics.

The fusion strategies live in :mod:`common.fusion` and the proxy metrics in
:mod:`common.metrics`; this module exposes the legacy registry interface used
across the pipeline:

* ``FUSION_METHODS`` -- name -> callable fusion instance (incl. ``LS_<name>``);
* ``best_metrics``   -- metric name -> True if a greater value is better;
* ``test_model_on_data`` -- evaluate every metric for a fused expert.

It also keeps the small helpers that are neither a fusion method nor a metric
(timing, random-MLP sanity checks, Qwen-module adaptation).
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
from common.metrics import METRICS, best_metrics
from common.names import TIME_METRIC
from common.numeric import _default_device

default_device = _default_device()

# name -> fusion instance (callable as ``method(mlp_a, mlp_b, train_data)`` or the
# k-agnostic ``method(*mlps, train_data=...)``). This *is* the fusion registry
# (an ordered, extensible Mapping), so the stages read methods only from the
# registry (§3.7) -- not from a literal list -- while staying a drop-in for the
# old ``name -> instance`` dict.
FUSION_METHODS = FUSION_REGISTRY


def count_mse(f, x, y):
    """Count MSE between f(x) and y."""
    return torch.mean((f(x) - y) ** 2)


def eval_merger(model1: MLP, model2: MLP, merged_model: MLP, data: torch.Tensor) -> torch.Tensor:
    """MSE of a fused model's reconstruction against both originals concatenated."""
    with torch.no_grad():
        return count_mse(
            merged_model.forward_unzip,
            data,
            torch.cat([model1.forward(data), model2.forward(data)], dim=-1),
        )


@torch.no_grad()
def find_mlp_mse_err(
    merger, n: int, hidden: int, out: int, examples: int = 10_000, device=default_device
) -> torch.Tensor:
    """Merge two random experts with ``merger`` and return reconstruction MSE.

    ``examples`` is the real number of train/test rows (no hidden multiplier).
    """
    with torch.no_grad():
        mlp1 = AdvancedMLP(n, hidden, out, device=device)
        mlp2 = AdvancedMLP(n, hidden, out, device=device)

        x_train = torch.rand((examples, n), device=device)
        x_test = torch.rand((examples, n), device=device)

        mlp_merged = merger(mlp1, mlp2, x_train)
        res = eval_merger(mlp1, mlp2, mlp_merged, x_test)

    del mlp_merged, x_train, x_test, mlp1, mlp2
    torch.cuda.empty_cache()
    return res


def find_mlp_mse_err_repeat(merger, n: int, hidden: int, out: int, repeat: int = 1000) -> torch.Tensor:
    """Repeat :func:`find_mlp_mse_err` and stack the results."""
    return (
        torch.stack(
            [find_mlp_mse_err(merger, n, hidden, out, examples=1000) for _ in range(repeat)],
            dim=0,
        )
        .detach()
        .cpu()
    )


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
def test_model_on_data(fused_mlp, mlp1, mlp2, test_data) -> dict[str, float]:
    """Evaluate every metric (plus wall-clock ``Time`` in ms) for a fused expert."""
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
        report = {metric.name: float("nan") for metric in METRICS}
    else:
        report = {metric.name: metric(real_res, new_res) for metric in METRICS}
    report[TIME_METRIC] = 1000 * timer(fused_mlp, all_tokens)
    return report
