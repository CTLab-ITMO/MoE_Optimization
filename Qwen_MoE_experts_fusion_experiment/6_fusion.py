#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[1]:


import torch
import torch.nn as nn
import copy
from matplotlib import pyplot as plt
from tqdm import tqdm
import time
import sys
from config import LAYERS, EXPERTS
import os

sys.path.append("..")
from common.mlp import MLP, AdvancedMLP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FUSION_METHODS = {}


# # Classes

# In[2]:


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


# # Utils

# In[3]:


def count_mse(f, x, y):
    """Count MSE if f(x) and y"""
    return torch.mean((f(x) - y) ** 2)


def eval_merger(
    model1: MLP, model2: MLP, merged_model: MLP, data: torch.Tensor
) -> torch.Tensor:
    """Count MSE of zipped model with two simple models"""
    with torch.no_grad():
        return count_mse(
            merged_model.forward_unzip,
            data,
            torch.cat([model1.forward(data), model2.forward(data)], dim=-1),
        )


# In[4]:


def zipMLP(
    mlpA: MLP, mlpB: MLP, train_data: torch.Tensor, create_new: bool = True
) -> MLP:
    """Take two MLP models, zipit using train data. Return zipped MLP"""
    with torch.no_grad():
        new_mlp = mlpA
        if create_new:
            new_mlp = copy.deepcopy(mlpA)
        new_mlp.zip_with(mlpB)
        new_mlp.train_with(train_data)
        new_mlp.zip()
    return new_mlp


def avgMLP(
    mlpA: MLP, mlpB: MLP, train_data: torch.Tensor, create_new: bool = True
) -> MLP:
    """Take two MLP models. Return same model with parameters averaged"""
    with torch.no_grad():
        new_mlp = mlpA
        if create_new:
            new_mlp = copy.deepcopy(mlpA)
        new_dict = dict()
        dictA = mlpA.state_dict()
        dictB = mlpB.state_dict()
        for key in dictA.keys():
            new_dict[key] = (dictA[key] + dictB[key]) / 2
        new_mlp.load_state_dict(new_dict)
    return new_mlp


def proj(*args):
    return args[0]


FUSION_METHODS["ZipIt"] = zipMLP
FUSION_METHODS["Avarage"] = avgMLP
FUSION_METHODS["Projection"] = proj


# In[5]:


def find_mlp_mse_err(
    merger, n: int, hidden: int, out: int, examples: int = 10_000
) -> torch.Tensor:
    """For a given merger make two random models, 
    merge them, train and test on random data. Return it's MSE with original models"""
    with torch.no_grad():
        mlp1 = AdvancedMLP(n, hidden, out)
        mlp2 = AdvancedMLP(n, hidden, out)

        x_train = torch.rand((10 * examples, n), device=device)
        x_test = torch.rand((10 * examples, n), device=device)

        mlp_merged = merger(mlp1, mlp2, x_train)

        res = eval_merger(mlp1, mlp2, mlp_merged, x_test)

    del mlp_merged, x_train, x_test, mlp1, mlp2
    torch.cuda.empty_cache()
    return res


# In[6]:


def find_mlp_mse_err_repeat(
    merger, n: int, hidden: int, out: int, repeat: int = 1000
) -> torch.Tensor:
    """Repeat previous test [repeat] times. Return a tensor of results"""
    return (
        torch.stack(
            [
                find_mlp_mse_err(merger, n, hidden, out, examples=1000)
                for _ in tqdm(range(repeat))
            ],
            dim=0,
        )
        .detach()
        .cpu()
    )


# In[7]:


def turn_in_mlp(mlp: nn.Module) -> MLP:
    """Extract matrixs from loaded mlp layer and create AdvancedMLP with same weights"""
    new_mlp = AdvancedMLP(
        mlp.gate_proj.in_features,
        mlp.gate_proj.out_features,
        mlp.down_proj.out_features,
    )
    new_mlp.gate = mlp.gate_proj
    new_mlp.up = mlp.up_proj
    new_mlp.down = mlp.down_proj
    return new_mlp


# In[8]:


def mse_on_test_data(mlp_zipped: MLP, mlp1: MLP, mlp2: MLP, test_data) -> float:
    """Apply default mlps and zipped version and return MSE between this values"""
    real_res = torch.cat(
        [
            torch.cat([mlp1.forward(example), mlp2.forward(example)], dim=-1)
            for example in test_data
        ]
    )
    return torch.mean(
        torch.square(
            torch.cat([mlp_zipped.forward_unzip(example) for example in test_data]) - real_res
        )
    ).item()


def timer(mlp: MLP, test_data, iters=10) -> float:
    """Apply mlp on data and return required time"""
    start_time = time.perf_counter()
    for _ in range(iters):
        for example in test_data:
            y = mlp.forward(example)
            if mlp.U is not None:
                torch._C._nn.linear(y, mlp.U)
    end_time = time.perf_counter()
    return end_time - start_time


# # Fusion

# In[9]:


def compare_fusions_error_and_time(
    model1, model2, train_data, test_data, ind=None
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Take two experts from model first layer and count MSE and time for existed fusion methods"""
    torch.cuda.empty_cache()
    mlp1 = turn_in_mlp(model1)
    mlp2 = turn_in_mlp(model2)
    fusions = {}
    for fusion_name, fusion_methos in FUSION_METHODS.items():
        fusions[fusion_name] = fusion_methos(mlp1, mlp2, train_data)
    if ind is not None:
        fusion_experts[ind] = fusions["ZipIt"]
    torch.cuda.empty_cache()

    means = {
        f"MSE of {fusion_name}:": mse_on_test_data(
            fusions[fusion_name], mlp1, mlp2, test_data
        )
        for fusion_name, _ in FUSION_METHODS.items()
    }

    times = {
        f"Time of {fusion_name}:": timer(fusions[fusion_name], test_data)
        for fusion_name, _ in FUSION_METHODS.items()
    }

    torch.cuda.empty_cache()
    return means, times


# In[10]:


for layer in tqdm(LAYERS):
    experts = [
        torch.load(f"data/experts/{layer}/{i}.pt", weights_only=False)
        for i in range(EXPERTS)
    ]
    train_data = torch.load(f"data/mlp_inputs/train_{layer}.pt")
    test_data = torch.load(f"data/mlp_inputs/test_{layer}.pt")
    train_activations = torch.load(f"data/activations/train_{layer}.pt")
    test_activations = torch.load(f"data/activations/test_{layer}.pt")
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
        experiment_results.append(
            compare_fusions_error_and_time(
                experts[expert_1].to(device),
                experts[expert_2].to(device),
                train_data_for_pair,
                test_data_for_pair,
                i,
            )
        )

        new_experiment_results = []
        for i in range(len(experiment_results[0])):
            tmp = {}
            for key in experiment_results[0][i].keys():
                tmp[key] = []
                for exper in experiment_results:
                    tmp[key].append(exper[i][key])
            new_experiment_results.append(tmp)

        os.makedirs(f"data/fusion_experts/{layer}", exist_ok=True)
        for i, expert in enumerate(fusion_experts):
            torch.save(expert, f"data/fusion_experts/{layer}/{i}.pt")


# In[11]:


for experiment in new_experiment_results:
    keys = list(experiment.keys())
    plt.boxplot([experiment[key] for key in keys])
    plt.legend(keys)
    plt.show()


# In[12]:


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
    plt.show()


# In[13]:


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
    plt.show()


# In[14]:


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
    plt.show()


# In[15]:


for experiment in new_experiment_results:
    keys = list(experiment.keys())
    plt.violinplot([experiment[key] for key in keys])
    plt.legend(keys)
    plt.show()


# In[16]:


for key, values in new_experiment_results[0].items():
    print(f"{key}'s mean value: {sum(values) / len(values)}")
print()
for key, values in new_experiment_results[1].items():
    print(f"{key}'s mean value: {sum(values) / len(values)}")

