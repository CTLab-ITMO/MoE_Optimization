import torch
from datetime import datetime
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
import sys
from config import (
    LAYERS,
    EXPERTS,
    device,
    DATA_DIR,
    FILTER_FUSIONS,
    FILTER_METRICS,
    MIN_PAIR_VALIDATION_SAMPLES,
)
import os
import seaborn as sns
import numpy as np

sys.path.append("..")
from common.mlp_utils import FUSION_METHODS, test_model_on_data, best_metrics

# Number of times stage 6 retries a flaky torch.save before giving up.
IO_SAVE_RETRIES = 3
# Metric values at or above this are treated as unstable outliers and dropped before plotting.
MAX_PLOT_METRIC_VALUE = 1e6

if FILTER_FUSIONS is not None:
    FUSION_METHODS = {k:v for k, v in FUSION_METHODS.items() if k in FILTER_FUSIONS}
if FILTER_METRICS is not None:
    best_metrics = {k:v for k, v in best_metrics.items() if k in FILTER_METRICS}


from common.seeding import set_seed
from config import SEED
set_seed(SEED)


@torch.no_grad()
def compare_fusions_error_and_time(
    mlp1, mlp2, train_data, validation_data, fusion_experts=None, ind=None
) -> dict[str, dict[str, list[float]]]:
    """Take two experts from model first layer
    and count MSE and time for existed fusion methods"""
    fusions = {}
    for fusion_name, fusion_methos in FUSION_METHODS.items():
        fusions[fusion_name] = fusion_methos(mlp1, mlp2, train_data)
    if fusion_experts is not None:
        fusion_experts[ind] = fusions

    for metric, higher_is_better in best_metrics.items():
        fusions[f"Best_{metric}"] = fusions["Projection"]
    if len(validation_data) > MIN_PAIR_VALIDATION_SAMPLES:
        results = {
            fusion_name: test_model_on_data(fusions[fusion_name], mlp1, mlp2, validation_data)
            for fusion_name, _ in FUSION_METHODS.items()
        }
        for metric, higher_is_better in best_metrics.items():
            best_name = list(results.keys())[0]
            for fusion_name in FUSION_METHODS.keys():
                if (results[fusion_name][metric] > results[best_name][metric]) == higher_is_better:
                    best_name = fusion_name
            fusions[f"Best_{metric}"] = fusions[best_name]
            results[f"Best_{metric}"] = results[best_name]

        return results
    else:
        return None


experiment_results: list[dict[str, dict[str, list[float]]]] = []
act_dir = f"{DATA_DIR}/activations"
for layer in tqdm(LAYERS):
    experts = [
        torch.load(f"{DATA_DIR}/experts/{layer}/expert_{i}.pt", weights_only=False)
        for i in range(EXPERTS)
    ]
    train_data = torch.load(f"{DATA_DIR}/mlp_inputs/train_{layer}.pt")
    validation_data = torch.load(f"{DATA_DIR}/mlp_inputs/validation_{layer}.pt")
    pairs = torch.load(f"{DATA_DIR}/pairs/{layer}.pt")
    train_pairs_activations = torch.load(f"{act_dir}/pairs_train_{layer}.pt")
    validation_pairs_activations = torch.load(f"{act_dir}/pairs_validation_{layer}.pt")

    fusion_experts = [None] * len(pairs)

    for i in tqdm(range(len(pairs))):
        expert_1, expert_2 = pairs[i]
        train_data_for_pair = [
            train_data[ind].to(device) for ind in train_pairs_activations[i]
        ]
        validation_data_for_pair = [
            validation_data[ind].to(device) for ind in validation_pairs_activations[i]
        ]
        res = compare_fusions_error_and_time(
            experts[expert_1].to(device),
            experts[expert_2].to(device),
            train_data_for_pair,
            validation_data_for_pair,
            fusion_experts,
            i,
        )
        if len(validation_data_for_pair) > MIN_PAIR_VALIDATION_SAMPLES:
            experiment_results.append(res)

    for x in train_data:
        del x
    for x in validation_data:
        del x
    for i, fused in enumerate(fusion_experts):
        for fusion_name, fusion_expert in fused.items():
            os.makedirs(f"{DATA_DIR}/fusion_experts/{fusion_name}/{layer}", exist_ok=True)
            for _ in range(IO_SAVE_RETRIES):  # Here sometimes were IO mistakes, try few times
                try:
                    torch.save(fusion_expert.cpu(), f"{DATA_DIR}/fusion_experts/{fusion_name}/{layer}/{i}.pt")
                    _io_err = None
                    break
                except Exception as e:
                    _io_err = e
                    print(f"[DEBUG] {e}")
            if _io_err is not None:
                raise _io_err
            del fusion_expert
    torch.cuda.empty_cache()


now = datetime.now()
timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
_plot_dir = f"plots/fusion/{timestamp}"
os.makedirs(_plot_dir, exist_ok=True)

os.makedirs(f"logs/fusion", exist_ok=True)
torch.save(experiment_results, f"logs/fusion/{timestamp}.pt")


experiment_results = torch.load(f"logs/fusion/{timestamp}.pt")


metrics = list(best_metrics.keys()) + ["Time"]
log_safe_metrics = {"MSE", "RMSE", "MAE", "RelMSE", "JS", "KL", "CKA", "EntropyGap", "Time"}
data = {    
    metric: {
        fusion_name: [experiment_result[fusion_name][metric] for experiment_result in experiment_results]
        for fusion_name in FUSION_METHODS.keys()
    } for metric in metrics
}


experts = len(data[metrics[0]][list(FUSION_METHODS.keys())[0]]) // len(LAYERS)

def _finite_median(values):
    """Median over finite values only (drops NaN/Inf from unstable fusions)."""
    finite = [value for value in values if np.isfinite(value)]
    return np.median(finite) if finite else float("nan")


for metric in metrics:
    print(f"Meadian {metric} results for fusion methods:")
    for method_name in FUSION_METHODS.keys():
        print(f"For {method_name} - {_finite_median(data[metric][method_name]):.3f}")

for i, layer in enumerate(LAYERS):
    print()
    print(f"On layer {layer}:")
    for metric in metrics:
        print(f"{metric} results for fusion methods:")
        for method_name in FUSION_METHODS.keys():
            print(f"For {method_name} - {_finite_median(data[metric][method_name][i*experts:(i+1)*experts]):.3f}")


for values in data.values():
    for key in values.keys():
        values[key] = [
            value for value in values[key]
            if np.isfinite(value) and value < MAX_PLOT_METRIC_VALUE
        ]


def _log_if_needed(values, metric):
    """Log-transform values for metrics whose spread reads better on a log scale."""
    if metric in log_safe_metrics and metric != "Cos":
        return np.log(np.maximum(np.asarray(values, dtype=np.float64), 1e-12))
    return values


def _plot_grid(filename, draw_panel):
    """Build a one-row subplot grid (one panel per metric), then save and close it."""
    plt.figure(figsize=(6 * len(metrics), 6))
    for i, metric in enumerate(metrics):
        plt.subplot(1, len(metrics), i + 1)
        plt.title(metric)
        draw_panel(metric)
    plt.savefig(f"{_plot_dir}/{filename}")
    plt.close()


def _hist_panel(metric, log_y):
    for name, values in data[metric].items():
        plt.hist(
            _log_if_needed(values, metric),
            label=name, bins=100, density=True, alpha=0.5,
            cumulative=True, histtype="stepfilled",
        )
    plt.legend()
    if log_y:
        plt.yscale("log")


def _violin_panel(metric):
    names = list(data[metric].keys())
    handles = []
    for j, name in enumerate(names):
        vp = plt.violinplot(_log_if_needed(data[metric][name], metric), positions=[j])
        handles.append(vp['bodies'][0])
    plt.legend(handles, names)


def _kde_panel(metric):
    names = list(data[metric].keys())
    sns.kdeplot([_log_if_needed(data[metric][key], metric) for key in names], fill=True, alpha=0.3)
    plt.legend(names)


_plot_grid("hist_stepfilled_1.png", lambda metric: _hist_panel(metric, log_y=False))
_plot_grid("hist_stepfilled_2.png", lambda metric: _hist_panel(metric, log_y=True))
_plot_grid("violinplot.png", _violin_panel)
_plot_grid("violinplot_log.png", _violin_panel)
_plot_grid("kdeplot.png", _kde_panel)

