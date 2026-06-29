"""Diagnostic plots of per-method fusion reconstruction-metric distributions.

Ported from the legacy stage-6 plotting block, with one structural change: every
output is written under an explicit ``out_dir`` (a ``DATA_DIR``-scoped plots
directory) instead of CWD-relative ``plots/`` (§1.4). matplotlib runs head-less
(Agg backend); the kde grid is skipped when seaborn is unavailable.

These are best-effort diagnostics: :func:`plot_fusion_metric_grids` renders each
grid independently and warns (rather than crashing) on a failing grid, and the
calling stage wraps the whole call so plotting can never break fusion.

Library code only -- this module never imports an experiment ``config``.
"""
import os
import warnings

import matplotlib

matplotlib.use("Agg")  # head-less rendering; no display needed

import numpy as np  # noqa: E402 - after the backend is fixed
from matplotlib import pyplot as plt  # noqa: E402

from common.metrics import log_safe_metrics  # noqa: E402
from common.names import TIME_METRIC  # noqa: E402

# The kde grid needs seaborn; probe once at import so the grid is cleanly skipped
# (not error-swallowed) when the optional dependency is absent.
try:
    import seaborn as _seaborn  # noqa: E402,F401 - availability probe only
    _HAS_SEABORN = True
except Exception:  # seaborn is an optional dependency
    _HAS_SEABORN = False

# Reconstruction metrics above this magnitude are unstable outliers (a fusion
# that blew up); clamp them out of the distribution plots so the axes stay
# readable. (Legacy ``MAX_PLOT_METRIC_VALUE``, §6.)
MAX_PLOT_METRIC_VALUE = 1e6

# Floor for the log-transform so log(0)/log(negative) never reaches matplotlib.
_LOG_FLOOR = 1e-12


def _finite_median(values) -> float:
    """Median over finite values only (drops NaN/Inf from unstable fusions)."""
    finite = [value for value in values if np.isfinite(value)]
    return float(np.median(finite)) if finite else float("nan")


def metric_medians(data: dict, metrics, method_names) -> dict:
    """Per-``(metric, method)`` finite median over the collected values.

    ``data`` maps ``metric -> {method_name -> [values across (layer, group)]}``.
    Returns ``{metric: {method_name: median}}`` -- the tabular summary the stage
    prints alongside the plots.
    """
    return {
        metric: {name: _finite_median(data[metric][name]) for name in method_names}
        for metric in metrics
    }


def _clamp(values, max_value) -> list:
    """Drop non-finite and outlier (``>= max_value``) values for readable axes."""
    return [value for value in values if np.isfinite(value) and value < max_value]


def _log_if_needed(values, metric, log_safe):
    """Log-transform values whose spread reads better on a log scale (§4.4)."""
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
    """Write per-metric distribution grids of the fusion methods under ``out_dir``.

    ``data`` maps ``metric -> {method_name -> [values across (layer, group)]}``;
    ``metrics`` is the ordered list of metric names to panel (one subplot column
    each). Every output is written under ``out_dir`` (a ``DATA_DIR``-scoped
    directory), never CWD-relative ``plots/`` (§1.4).

    Each grid is rendered independently: a failure in one grid warns and is
    skipped so the others still render. Returns the list of written file paths.
    """
    if log_safe is None:
        # ``Time`` is plotted on a log axis too (legacy parity), so add it to the
        # log_scale-derived metric set (§4.4) rather than re-listing literals.
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
                # One labeled kde per method (not a wide-form list, which seaborn
                # would mis-group by column index).
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
