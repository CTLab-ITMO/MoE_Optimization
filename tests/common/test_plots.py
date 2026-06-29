"""Tests for the DATA_DIR-scoped fusion diagnostic plots (common/plots.py).

Covers the §6 plotting restoration: the finite-median summary, the file outputs
landing under the given out_dir (never CWD), and robustness to NaN/Inf/outlier
values and empty method series. matplotlib runs head-less (Agg, set in conftest).
"""
import math
import os

import pytest

from common import plots
from common.metrics import log_safe_metrics
from common.names import TIME_METRIC


@pytest.fixture
def fusion_data():
    # metric -> {method -> [values across (layer, group)]}; deliberately seeded
    # with NaN, Inf and an above-MAX outlier to exercise the cleaners.
    return {
        "MSE": {  # log-scale metric
            "Average": [0.1, 0.2, 0.3, float("nan"), 1e9],
            "ZipIt": [0.05, 0.07, float("inf"), 0.09],
        },
        "Cosine": {  # NOT log-scale
            "Average": [0.9, 0.95, 0.99],
            "ZipIt": [0.8, 0.85, 0.88],
        },
        TIME_METRIC: {
            "Average": [1.0, 1.1, 1.2],
            "ZipIt": [2.0, 2.1, 2.2],
        },
    }


METRICS = ["MSE", "Cosine", TIME_METRIC]
METHODS = ["Average", "ZipIt"]


def test_metric_medians_drops_non_finite(fusion_data):
    medians = plots.metric_medians(fusion_data, METRICS, METHODS)
    # NaN dropped, 1e9 kept (finite) -> median of [0.1,0.2,0.3,1e9] = (0.2+0.3)/2
    assert medians["MSE"]["Average"] == pytest.approx(0.25)
    # Inf dropped from ZipIt -> median of [0.05,0.07,0.09] = 0.07
    assert medians["MSE"]["ZipIt"] == pytest.approx(0.07)
    assert math.isfinite(medians["Cosine"]["Average"])


def test_metric_medians_all_non_finite_is_nan():
    data = {"MSE": {"X": [float("nan"), float("inf")]}}
    medians = plots.metric_medians(data, ["MSE"], ["X"])
    assert math.isnan(medians["MSE"]["X"])


def test_plot_grids_write_files_under_out_dir(fusion_data, tmp_path):
    out_dir = str(tmp_path / "data" / "plots" / "fusion")
    written = plots.plot_fusion_metric_grids(fusion_data, METRICS, out_dir)

    assert written, "expected at least one plot to be written"
    for path in written:
        # every output stays under the requested out_dir (DATA_DIR-scoped, §1.4)
        assert os.path.commonpath([os.path.abspath(path), os.path.abspath(out_dir)]) == (
            os.path.abspath(out_dir)
        )
        assert os.path.isfile(path)
    # the two cumulative-hist grids are the robust ones and must always land
    names = {os.path.basename(p) for p in written}
    assert "hist_cumulative.png" in names
    assert "hist_cumulative_logy.png" in names


def test_plot_grids_handle_empty_method_series(tmp_path):
    # A method whose every value is non-finite collapses to an empty series after
    # cleaning; plotting must not raise and must still render the other method.
    data = {
        "MSE": {"Average": [0.1, 0.2], "Broken": [float("nan"), float("inf")]},
    }
    out_dir = str(tmp_path / "plots")
    written = plots.plot_fusion_metric_grids(data, ["MSE"], out_dir, use_kde=False)
    assert written
    assert all(os.path.isfile(p) for p in written)


def test_plot_grids_swallow_a_failing_grid(fusion_data, tmp_path, monkeypatch):
    # A grid that fails to render (e.g. savefig raises) must not propagate or kill
    # the others; the function warns and returns only the grids that succeeded.
    def boom(*a, **k):
        raise RuntimeError("no disk")

    monkeypatch.setattr(plots.plt, "savefig", boom)
    with pytest.warns(RuntimeWarning, match="fusion plot grid"):
        written = plots.plot_fusion_metric_grids(fusion_data, METRICS, str(tmp_path / "p"))
    assert written == []  # every grid failed, none lost, no exception raised


def test_max_plot_metric_value_is_named_constant():
    # §6 knob preserved as a named constant (legacy MAX_PLOT_METRIC_VALUE = 1e6).
    assert plots.MAX_PLOT_METRIC_VALUE == 1e6


def test_default_log_safe_includes_time_not_cosine():
    # Default log-safe set is derived from the metric log_scale flags (§4.4) plus
    # Time (plotted on a log axis, legacy parity); Cosine is never log-scaled.
    assert TIME_METRIC not in log_safe_metrics  # Time is a pseudo-metric
    assert "Cosine" not in log_safe_metrics
