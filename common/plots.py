"""Diagnostic distribution plots comparing fusion methods across metrics.

Writes all output to ``out_dir`` (never CWD-relative). Uses matplotlib Agg
(headless). KDE grids require seaborn; they are skipped when it is absent.
Each grid fails independently with a warning so a bad plot never breaks fusion.
Does not import experiment config.
"""
import os
import warnings

import matplotlib

matplotlib.use("Agg")  # head-less rendering; no display needed

import numpy as np  # noqa: E402 - after the backend is fixed
from matplotlib import pyplot as plt  # noqa: E402

from common.metrics import log_safe_metrics  # noqa: E402
from common.names import TIME_METRIC  # noqa: E402

# Probe seaborn once at import; grid is skipped cleanly when it is absent.
try:
    import seaborn as _seaborn  # noqa: E402,F401 - availability probe only
    _HAS_SEABORN = True
except Exception:  # seaborn is an optional dependency
    _HAS_SEABORN = False

# Values at or above this are considered blown-up outliers; clamped before plotting.
MAX_PLOT_METRIC_VALUE = 1e6

# Floor for the log-transform so log(0)/log(negative) never reaches matplotlib.
_LOG_FLOOR = 1e-12


def _finite_median(values) -> float:
    """Median over finite values only (drops NaN/Inf from unstable fusions)."""
    finite = [value for value in values if np.isfinite(value)]
    return float(np.median(finite)) if finite else float("nan")


def metric_medians(data: dict, metrics, method_names) -> dict:
    """Compute finite median per (metric, method) pair.

    ``data``: ``metric -> {method_name -> [values across (layer, group)]}``.
    Returns ``{metric: {method_name: median}}``.
    """
    return {
        metric: {name: _finite_median(data[metric][name]) for name in method_names}
        for metric in metrics
    }


def _clamp(values, max_value) -> list:
    """Drop non-finite and outlier (``>= max_value``) values for readable axes."""
    return [value for value in values if np.isfinite(value) and value < max_value]


def _log_if_needed(values, metric, log_safe):
    """Log-transform values whose spread reads better on a log scale."""
    if metric in log_safe:
        return np.log(np.maximum(np.asarray(values, dtype=np.float64), _LOG_FLOOR))
    return values


def plot_fusion_metric_grids(
    data: dict,
    metrics,
    out_dir: str,
    *,
    max_value: float = MAX_PLOT_METRIC_VALUE,
    log_safe=None,
    use_kde: bool = True,
) -> list:
    """Save distribution grids (hist, violin, optional kde) for each metric to ``out_dir``.

    ``data``: ``metric -> {method_name -> [values across (layer, group)]}``.
    Each grid (one PNG per plot type) is rendered independently; failures warn
    and are skipped. Returns paths of files actually written.
    """
    if log_safe is None:
        # Also log-transform Time without re-listing it as a literal.
        log_safe = set(log_safe_metrics) | {TIME_METRIC}
    os.makedirs(out_dir, exist_ok=True)

    # Clamp only the paneled metrics (``data`` may carry extras we never draw).
    clean = {
        metric: {
            name: _clamp(values, max_value)
            for name, values in data.get(metric, {}).items()
        }
        for metric in metrics
    }

    def _names_with_data(metric):
        """Methods whose series is non-empty after cleaning (the only plottable ones)."""
        return [name for name, values in clean[metric].items() if values]

    def _series(metric, name):
        return _log_if_needed(clean[metric][name], metric, log_safe)

    def _grid(filename, draw_panel):
        path = os.path.join(out_dir, filename)
        try:
            plt.figure(figsize=(6 * len(metrics), 6))
            for i, metric in enumerate(metrics):
                plt.subplot(1, len(metrics), i + 1)
                plt.title(metric)
                draw_panel(metric)
            plt.savefig(path)
            return path
        except Exception as exc:  # one bad grid must not lose the others
            warnings.warn(
                f"fusion plot grid {filename!r} failed: {exc!r}",
                RuntimeWarning,
                stacklevel=2,
            )
            return None
        finally:
            plt.close("all")

    def _hist_panel(metric, log_y):
        names = _names_with_data(metric)
        for name in names:
            plt.hist(
                _series(metric, name),
                label=name, bins=100, density=True, alpha=0.5,
                cumulative=True, histtype="stepfilled",
            )
        if names:
            plt.legend()
        if log_y:
            plt.yscale("log")

    def _violin_panel(metric):
        handles, labels = [], []
        for name in _names_with_data(metric):
            try:
                vp = plt.violinplot(_series(metric, name), positions=[len(labels)])
            except Exception:  # degenerate series (single point / zero variance)
                continue
            handles.append(vp["bodies"][0])
            labels.append(name)
        if handles:
            plt.legend(handles, labels)

    def _kde_panel(metric):
        import seaborn as sns  # guaranteed available (use_kde gated on _HAS_SEABORN)

        drawn = False
        for name in _names_with_data(metric):
            try:
                # Plot each method separately so seaborn labels them correctly.
                sns.kdeplot(_series(metric, name), label=name, fill=True, alpha=0.3)
                drawn = True
            except Exception:  # degenerate series -- skip this method's curve
                continue
        if drawn:
            plt.legend()

    grids = [
        ("hist_cumulative.png", lambda metric: _hist_panel(metric, log_y=False)),
        ("hist_cumulative_logy.png", lambda metric: _hist_panel(metric, log_y=True)),
        ("violin.png", _violin_panel),
    ]
    if use_kde and _HAS_SEABORN:
        grids.append(("kde.png", _kde_panel))

    written = []
    for filename, panel in grids:
        path = _grid(filename, panel)
        if path is not None:
            written.append(path)
    return written
