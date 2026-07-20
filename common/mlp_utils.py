"""Glue utilities and re-exports for the fusion pipeline.

Exposes ``FUSION_METHODS`` (name -> callable fusion instance, incl. ``LS_<name>``),
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
from common.fused_group import (
    RECONSTRUCTION_REGISTRY,
    default_reconstruction_for_size,
)
from common.metrics import METRIC_REGISTRY, best_metrics
from common.names import BEST_PREFIX, TIME_METRIC
from common.numeric import _default_device

default_device = _default_device()

# Alias for FUSION_REGISTRY: name -> fusion instance.
# Callable as method(*mlps, train_data=...); stages read methods only from this registry.
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
    """Merge two random AdvancedMLP experts with ``merger`` and return reconstruction MSE on ``examples`` rows."""
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
def reconstruct_members(fused_mlp, group_size: int, x: torch.Tensor) -> torch.Tensor:
    """Concatenated per-member reconstruction of a fused expert.

    Registry-driven: the size default (:func:`default_reconstruction_for_size`)
    resolves the reconstruction kind for ``group_size`` -- ``zip_unmerge`` for a
    pair (one ``forward_unzip`` call covering both members, the historical
    pairwise path kept byte-identical), ``identity`` for any other size (the
    fused output serves every member as-is, tiled ``group_size`` times) -- and
    that kind's ``reconstruct_all`` produces the concatenated output.
    """
    kind = default_reconstruction_for_size(group_size)
    spec = RECONSTRUCTION_REGISTRY[kind]
    if spec.reconstruct_all is None:
        raise NotImplementedError(
            f"reconstruction kind {kind!r} has no reconstruct_all; cannot score "
            f"a group of size {group_size}"
        )
    return spec.reconstruct_all(fused_mlp, group_size, x)


@torch.no_grad()
def timer(mlp: MLP, test_data, iters=3, *, group_size: int = 2) -> float:
    """Time the fused expert's reconstruction on data and return elapsed seconds.

    ``group_size=2`` (default) times ``mlp.forward_unzip`` as before; other
    sizes time the identity reconstruction (see :func:`reconstruct_members`).
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    for _ in range(iters):
        reconstruct_members(mlp, group_size, test_data)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    return end_time - start_time


@torch.no_grad()
def test_model_on_data(fused_mlp, *args, ctx=None) -> dict[str, float]:
    """Compute all registered metrics plus wall-clock ``Time`` (ms) for a fused expert.

    Call as ``test_model_on_data(fused, *members, test_data)``: the last
    positional is the evaluation data, everything before it the group's member
    experts (``k >= 1``; the historical pairwise call keeps working unchanged).
    The reference side concatenates every member's output; the fused side is
    the concatenated per-member reconstruction (:func:`reconstruct_members`).
    ``ctx`` is passed to each metric. New metrics are included automatically.
    """
    if len(args) < 2:
        raise TypeError(
            "test_model_on_data(fused, *members, test_data) needs at least "
            "one member expert and the evaluation data"
        )
    *members, test_data = args
    all_tokens = as_tensor(test_data)
    real_res = torch.cat([m.forward(all_tokens) for m in members], dim=-1).float()
    new_res = reconstruct_members(fused_mlp, len(members), all_tokens).float()

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
    report[TIME_METRIC] = 1000 * timer(fused_mlp, all_tokens, group_size=len(members))
    return report


def select_best_fused(fused_by_method, results_by_method, best_metrics, *, method_names=None):
    """Return ``{Best_<metric>: best_fused_expert}`` for each tracked metric.

    ``fused_by_method``: method name -> fused expert.
    ``results_by_method``: method name -> metric report from
    :func:`test_model_on_data`, or ``None`` if too few validation rows to score.
    Falls back to the first registered method when unscored; otherwise picks
    the winner per metric using its ``higher_is_better`` flag (first-method tie-break).
    """
    names = list(method_names) if method_names is not None else list(fused_by_method)
    fallback = names[0]
    chosen = {f"{BEST_PREFIX}{metric}": fused_by_method[fallback] for metric in best_metrics}
    if results_by_method is None:
        return chosen
    for metric, higher_is_better in best_metrics.items():
        best_name = names[0]
        for name in names:
            if (results_by_method[name][metric] > results_by_method[best_name][metric]) == higher_is_better:
                best_name = name
        chosen[f"{BEST_PREFIX}{metric}"] = fused_by_method[best_name]
    return chosen
