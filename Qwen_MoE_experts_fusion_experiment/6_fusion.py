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
from config import LAYERS, EXPERTS, device
import os
import seaborn as sns

sys.path.append("..")
from common.mlp import MLP, AdvancedMLP
from common.mlp_utils import FUSION_METHODS, turn_in_mlp, mse_on_test_data, timer


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

def compare_fusions_error_and_time(
    model1, model2, train_data, test_data, ind=None
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Take two experts from model first layer 
    and count MSE and time for existed fusion methods"""
    torch.cuda.empty_cache()
    mlp1 = turn_in_mlp(model1)
    mlp2 = turn_in_mlp(model2)
    fusions = {}
    for fusion_name, fusion_methos in FUSION_METHODS.items():
        fusions[fusion_name] = fusion_methos(mlp1, mlp2, train_data)
    if ind is not None:
        fusion_experts[ind] = fusions["ZipIt"]
    torch.cuda.empty_cache()

    if len(test_data_for_pair) > 0:
        means = {
            f"MSE of {fusion_name}": mse_on_test_data(
                fusions[fusion_name], mlp1, mlp2, test_data
            )
            for fusion_name, _ in FUSION_METHODS.items()
        }

        times = {
            f"Time of {fusion_name}": timer(fusions[fusion_name], test_data)
            for fusion_name, _ in FUSION_METHODS.items()
        }

        torch.cuda.empty_cache()
        return means, times
    else:
        return None, None


act_dir = "data/activations"
for layer in tqdm(LAYERS):
    experts = [
        torch.load(f"data/experts/{layer}/expert_{i}.pt", weights_only=False)
        for i in range(EXPERTS)
    ]
    train_data = torch.load(f"data/mlp_inputs/train_{layer}.pt")
    test_data = torch.load(f"data/mlp_inputs/test_{layer}.pt")
    pairs = torch.load(f"data/pairs_{layer}.pt")
    train_pairs_activations = torch.load(f"data/activations/pairs_train_{layer}.pt")
    test_pairs_activations = torch.load(f"data/activations/pairs_test_{layer}.pt")

    fusion_experts = [None] * len(pairs)

    experiment_results = []
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
            i,
        )
        if len(test_data_for_pair) > 0:
            experiment_results.append(res)

        os.makedirs(f"data/fusion_experts/{layer}", exist_ok=True)
        for i, expert in enumerate(fusion_experts):
            torch.save(expert, f"data/fusion_experts/{layer}/{i}.pt")


new_experiment_results = []
for j in range(len(experiment_results[0])):
    tmp = {}
    for key in experiment_results[0][j].keys():
        tmp[key] = []
        for exper in experiment_results:
            tmp[key].append(exper[j][key])
    new_experiment_results.append(tmp)


now = datetime.now()
time = now.strftime("%Y_%m_%d_%HX_%M_%S")
_plot_dir = f"plots/fusion/{time}"
os.makedirs(_plot_dir)


for experiment in new_experiment_results:
    keys = list(experiment.keys())
    plt.boxplot([experiment[key] for key in keys])
    plt.legend(keys)

    plt.savefig(f"{_plot_dir}/boxplot.png")
    plt.close()


for experiment in new_experiment_results:
    keys = list(experiment.keys())
    plt.hist(
        [experiment[key] for key in keys],
        cumulative=True,
        bins=100,
        density=True,
        histtype="stepfilled",
        alpha=0.5,
    )
    plt.legend(list(reversed(keys)))

    plt.savefig(f"{_plot_dir}/hist_stepfilled.png")
    plt.close()


for experiment in new_experiment_results:
    keys = list(experiment.keys())
    plt.hist(
        [experiment[key] for key in keys],
        cumulative=True,
        bins=100,
        density=True,
        histtype="step",
        alpha=0.5,
    )
    plt.legend(list(reversed(keys)))

    plt.savefig(f"{_plot_dir}/hist_step.png")
    plt.close()


for experiment in new_experiment_results:
    keys = list(experiment.keys())
    for key in keys:
        plt.hist(
            experiment[key],
            bins=30,
            density=True,
            alpha=0.5,
        )
    plt.legend(list(reversed(keys)))

    plt.savefig(f"{_plot_dir}/hist.png")
    plt.close()


for experiment in new_experiment_results:
    keys = list(experiment.keys())
    plt.violinplot([experiment[key] for key in keys])
    plt.legend(keys)

    plt.savefig(f"{_plot_dir}/violinplot.png")
    plt.close()


for experiment in new_experiment_results:
    keys = list(experiment.keys())
    sns.kdeplot([experiment[key] for key in keys], fill=True, alpha=0.3)
    plt.legend(keys)

    plt.savefig(f"{_plot_dir}/kdeplot.png")
    plt.close()


for key, values in new_experiment_results[0].items():
    print(f"{key}'s mean value: {sum(values) / len(values)}")
print()
for key, values in new_experiment_results[1].items():
    print(f"{key}'s mean value: {sum(values) / len(values)}")


list1 = new_experiment_results[0]["MSE of ZipIt"]
list2 = new_experiment_results[0]["MSE of Avarage"]
list3 = new_experiment_results[0]["MSE of Projection"]

plt.figure(figsize=(10, 6))
sns.kdeplot(list1, label='Our method', fill=True, alpha=0.3)
sns.kdeplot(list2, label='Average', fill=True, alpha=0.3)
sns.kdeplot(list3, label='Projection', fill=True, alpha=0.3)
plt.xlabel('MSE')
plt.ylabel('Density')
plt.title('Density of errors')
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig(f"{_plot_dir}/kde_errors.png")
plt.close()


list1 = new_experiment_results[1]["Time of ZipIt"]
list2 = new_experiment_results[1]["Time of Avarage"]
list3 = new_experiment_results[1]["Time of Projection"]

plt.figure(figsize=(10, 6))
sns.kdeplot(list1, label='Our method', fill=True, alpha=0.3)
sns.kdeplot(list2, label='Avarage весов', fill=True, alpha=0.3)
sns.kdeplot(list3, label='Projection', fill=True, alpha=0.3)
plt.xlabel('Time')
plt.ylabel('Density')
plt.title('Density of comptation time')
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig(f"{_plot_dir}/kde_time.png")
plt.close()

