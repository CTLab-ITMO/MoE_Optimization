#!/usr/bin/env python
# coding: utf-8

# # Imports

import torch
from datetime import datetime
import torch.nn as nn
import copy
from matplotlib import pyplot as plt
from tqdm import tqdm
import time
import sys
from config import LAYERS, EXPERTS, device, DATA_DIR
import os
import seaborn as sns
import numpy as np

sys.path.append("..")
from common.mlp import MLP, AdvancedMLP
from common.mlp_utils import FUSION_METHODS, turn_in_mlp, mse_on_test_data, timer, cosine_on_test_data


# # Classes

class Logger(nn.Module):
    """Simple class for saving passing values"""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.xs: list[torch.Tensor] | None = None

    def forward(self, x: torch.Tensor):
        if self.xs is not None:
            self.xs.append(x)
        return self.model(x)


# # Fusion

@torch.autocast("cuda")
@torch.no_grad()
def compare_fusions_error_and_time(
    mlp1, mlp2, train_data, test_data, fusion_experts=None, ind=None
) -> dict[str, dict[str, list[float]]]:
    """Take two experts from model first layer 
    and count MSE and time for existed fusion methods"""
    fusions = {}
    for fusion_name, fusion_methos in FUSION_METHODS.items():
        fusions[fusion_name] = fusion_methos(mlp1, mlp2, train_data)
    if fusion_experts is not None:
        fusion_experts[ind] = fusions

    fusions["Best"] = fusions.get("Avarage", list(fusions.values())[0])
    if len(test_data) > 10:
        results = {
            fusion_name: {
                "MSE": mse_on_test_data(fusions[fusion_name], mlp1, mlp2, test_data),
                "Cosine": cosine_on_test_data(fusions[fusion_name], mlp1, mlp2, test_data),
                "Time": timer(fusions[fusion_name], test_data)
            }
            for fusion_name, _ in FUSION_METHODS.items()
        }
        best_name = list(results.keys())[0]
        for fusion_name, result in results.items():
            if result["MSE"] < results[best_name]["MSE"]:
                best_name = fusion_name
        fusions["Best"] = fusions[best_name]

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
    test_data = torch.load(f"{DATA_DIR}/mlp_inputs/test_{layer}.pt")
    pairs = torch.load(f"{DATA_DIR}/pairs/{layer}.pt")
    train_pairs_activations = torch.load(f"{DATA_DIR}/activations/pairs_train_{layer}.pt")
    test_pairs_activations = torch.load(f"{DATA_DIR}/activations/pairs_test_{layer}.pt")

    fusion_experts = [None] * len(pairs)

    for i in tqdm(range(len(pairs))):
        expert_1, expert_2 = pairs[i]
        train_data_for_pair = [
            train_data[ind].to(device) for ind in train_pairs_activations[i]
        ]
        test_data_for_pair = [
            test_data[ind].to(device) for ind in test_pairs_activations[i]
        ]
        res = compare_fusions_error_and_time(
            experts[expert_1].to(device),
            experts[expert_2].to(device),
            train_data_for_pair,
            test_data_for_pair,
            fusion_experts,
            i,
        )
        if len(test_data_for_pair) > 10:
            experiment_results.append(res)

    for x in train_data:
        del x
    for x in test_data:
        del x
    for i, fused in enumerate(fusion_experts):
        for fusion_name, fusion_expert in fused.items():
            os.makedirs(f"{DATA_DIR}/fusion_experts/{fusion_name}/{layer}", exist_ok=True)
            torch.save(fusion_expert.cpu(), f"{DATA_DIR}/fusion_experts/{fusion_name}/{layer}/{i}.pt")
            del fusion_expert
    torch.cuda.empty_cache()


now = datetime.now()
time = now.strftime("%Y_%m_%d_%H_%M_%S")
_plot_dir = f"plots/fusion/{time}"
os.makedirs(_plot_dir, exist_ok=True)

os.makedirs(f"logs/fusion", exist_ok=True)
torch.save(experiment_results, f"logs/fusion/{time}.pt")


experiment_results = torch.load(f"logs/fusion/{time}.pt")


metrics = ["MSE", "Time", "Cosine"]
data = {    
    metric: {
        fusion_name: [experiment_result[fusion_name][metric] for experiment_result in experiment_results]
        for fusion_name in FUSION_METHODS.keys()
    } for metric in metrics
}


experts = len(data[metrics[0]][list(FUSION_METHODS.keys())[0]]) // len(LAYERS)

for metric in metrics:
    print(f"Mean {metric} results for fusion methods:")
    for method_name in FUSION_METHODS.keys():
        print(f"For {method_name} - {np.mean(data[metric][method_name]):.3f}")

for i, layer in enumerate(LAYERS):
    print()
    print(f"On layer {layer}:")
    for metric in metrics:
        print(f"{metric} results for fusion methods:")
        for method_name in FUSION_METHODS.keys():
            print(f"For {method_name} - {np.mean(data[metric][method_name][i*experts:(i+1)*experts]):.3f}")


plt.figure(figsize=(6 * len(metrics), 6))
for i, metric in enumerate(metrics):
    plt.subplot(1, len(metrics), i+1)
    for name, values in data[metric].items():
        plt.hist(values if metric == "Cosine" else np.log(values), label=name, bins=100, density=True, alpha=0.5, cumulative=True, histtype="stepfilled")
    plt.legend()
    plt.title(metric)
plt.savefig(f"{_plot_dir}/hist_stepfilled_1.png")
plt.close()


plt.figure(figsize=(6 * len(metrics), 6))
for i, metric in enumerate(metrics):
    plt.subplot(1, len(metrics), i+1)
    for name, values in data[metric].items():
        plt.hist(values, label=name, bins=100, density=True, alpha=0.5, cumulative=True, histtype="stepfilled")
    plt.legend()
    plt.title(metric)
plt.savefig(f"{_plot_dir}/hist_stepfilled_2.png")
plt.close()


plt.figure(figsize=(6 * len(metrics), 6))
for i, metric in enumerate(metrics):
    plt.subplot(1, len(metrics), i+1)
    names = list(data[metric].keys())
    for i, name in enumerate(names):
        vp = plt.violinplot(data[metric][name], positions=[i])
    l = []
    for x in names:
        l.extend([x, None, None, None])
    plt.legend(l)
    plt.title(metric)
plt.savefig(f"{_plot_dir}/violinplot.png")
plt.close()


plt.figure(figsize=(6 * len(metrics), 6))
for i, metric in enumerate(metrics):
    plt.subplot(1, len(metrics), i+1)
    names = list(data[metric].keys())
    sns.kdeplot([np.log(data[metric][key]) for key in names], fill=True, alpha=0.3)
    plt.legend(names)
plt.savefig(f"{_plot_dir}/kdeplot.png")
plt.close()

