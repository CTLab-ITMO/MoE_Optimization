"""Tests for the glue utilities and registries in ``common.mlp_utils``.

Fusion-method behavior lives in ``test_fusion.py`` and metric correctness in
``test_metrics.py``; this file covers only what ``mlp_utils`` itself owns:
timing, random-MLP sanity checks, Qwen-module adaptation, the combined
``test_model_on_data`` report, and the backward-compatible registries.
"""

import pytest
import torch
from torch import nn

from common import mlp_utils
from common.mlp import AdvancedMLP

CPU = torch.device("cpu")


@pytest.fixture
def pair(make_advanced_mlp):
    a = make_advanced_mlp(4, 6, 4, seed=1)
    b = make_advanced_mlp(4, 6, 4, seed=2)
    return a, b


@pytest.fixture
def data():
    torch.manual_seed(0)
    return torch.randn(32, 4)


def test_count_mse_zero_for_identity():
    x = torch.randn(4, 3)
    assert mlp_utils.count_mse(lambda v: v, x, x).item() == pytest.approx(0.0)


def test_eval_merger_returns_scalar(pair, data):
    a, b = pair
    merged = mlp_utils.FUSION_METHODS["Average"](a, b, data)
    err = mlp_utils.eval_merger(a, b, merged, data)
    assert err.dim() == 0
    assert err.item() >= 0.0


def test_find_mlp_mse_err_runs():
    err = mlp_utils.find_mlp_mse_err(
        mlp_utils.FUSION_METHODS["Average"], n=4, hidden=6, out=4, examples=2, device=CPU
    )
    assert err.item() >= 0.0


def test_find_mlp_mse_err_uses_examples_as_real_row_count():
    # §4.5: ``examples`` is the real number of rows -- no hidden ``10 *`` factor.
    seen = {}

    def capturing_merger(mlp1, mlp2, x_train):
        seen["rows"] = x_train.shape[0]
        return mlp1

    mlp_utils.find_mlp_mse_err(capturing_merger, n=4, hidden=6, out=4, examples=7, device=CPU)
    assert seen["rows"] == 7


def test_find_mlp_mse_err_repeat_stacks_results(monkeypatch):
    monkeypatch.setattr(
        mlp_utils, "find_mlp_mse_err", lambda *a, **k: torch.tensor(0.5)
    )
    out = mlp_utils.find_mlp_mse_err_repeat(
        mlp_utils.FUSION_METHODS["Average"], 4, 6, 4, repeat=3
    )
    assert out.shape == (3,)


def test_turn_in_mlp_wraps_qwen_style_module():
    class FakeQwenMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(4, 6, bias=False)
            self.up_proj = nn.Linear(4, 6, bias=False)
            self.down_proj = nn.Linear(6, 4, bias=False)

    fake = FakeQwenMLP()
    mlp = mlp_utils.turn_in_mlp(fake, device=CPU)
    assert isinstance(mlp, AdvancedMLP)
    assert mlp.gate is fake.gate_proj
    assert mlp.forward(torch.randn(3, 4)).shape == (3, 4)


def test_timer_returns_nonnegative_duration(pair, data):
    a, b = pair
    fused = mlp_utils.FUSION_METHODS["Average"](a, b, data)
    elapsed = mlp_utils.timer(fused, data, iters=2)
    assert elapsed >= 0.0


def test_test_model_on_data_reports_all_metrics(pair, data):
    a, b = pair
    fused = mlp_utils.FUSION_METHODS["Average"](a, b, data)
    report = mlp_utils.test_model_on_data(fused, a, b, data)
    expected_keys = {
        "MSE", "RMSE", "MAE", "RelMSE", "Cosine", "Pearson", "R2",
        "CKA", "JS", "KL", "Top1", "Top5", "EntropyGap", "Time",
    }
    assert set(report) == expected_keys
    for value in report.values():
        assert isinstance(value, float)


def test_test_model_on_data_warns_and_nans_on_nonfinite(pair, data):
    """A non-finite fused output warns (not raises) and yields NaN metrics so the
    evaluation is dropped during averaging; ``Time`` stays a real measurement."""
    import math

    a, b = pair

    class NanFused:
        def forward_unzip(self, x, out=None):
            return torch.full((x.shape[0], 2 * x.shape[1]), float("nan"))

    with pytest.warns(RuntimeWarning):
        report = mlp_utils.test_model_on_data(NanFused(), a, b, data)

    metric_keys = set(report) - {"Time"}
    assert metric_keys == {
        "MSE", "RMSE", "MAE", "RelMSE", "Cosine", "Pearson", "R2",
        "CKA", "JS", "KL", "Top1", "Top5", "EntropyGap",
    }
    assert all(math.isnan(report[key]) for key in metric_keys)
    assert math.isfinite(report["Time"])


def test_fusion_methods_registry_complete():
    expected_base = {
        "ZipIt", "Average", "Pruning", "Procrustes", "Fisher", "TIES", "Distillation",
    }
    assert expected_base <= set(mlp_utils.FUSION_METHODS)
    assert "Projection" not in mlp_utils.FUSION_METHODS
    for name in expected_base:
        assert f"LS_{name}" in mlp_utils.FUSION_METHODS
    for method in mlp_utils.FUSION_METHODS.values():
        assert callable(method)


def test_best_metrics_direction_flags():
    assert mlp_utils.best_metrics["MSE"] is False
    assert mlp_utils.best_metrics["Cosine"] is True
    assert set(mlp_utils.best_metrics).issuperset({"R2", "KL", "Top1", "CKA"})
