#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import os
from tqdm import tqdm
import transformers
import datasets
from config import (
    MODEL_NAME,
    TRAIN_DATASET_NAME,
    TEST_DATASET_NAME,
    TRAIN_SIZE,
    TEST_SIZE,
    LAYERS,
    TOP_K,
    device,
    DATA_DIR,
)

print("Device:", device)


import sys

sys.path.append("..")
from common.seeding import set_seed
from config import SEED

set_seed(SEED)


if "model" not in globals():
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map=device
    )
if "tokenizer" not in globals():
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)

# Remove extra layers
MAX_LAYERS = max(LAYERS) + 1
for i in range(MAX_LAYERS, len(model.model.layers)):
    del model.model.layers[MAX_LAYERS]

torch.cuda.empty_cache()
model = model.to(device)


if "_train_data" not in globals():
    _train_data = datasets.load_dataset(**TRAIN_DATASET_NAME)["question"]
if "_test_data" not in globals():
    _test_data = datasets.load_dataset(**TEST_DATASET_NAME)["question"]
train_data = _train_data[:TRAIN_SIZE]
test_data = _test_data[:TEST_SIZE]
print("Example of train data:", train_data[0])
print("Example of test data:", test_data[0])


class ExpertsLogger(nn.Module):
    """Simple class for saving activated experts based on gate"""

    def __init__(self, gate: nn.Module):
        super().__init__()
        self.gate = gate
        self.num_experts = gate.num_experts
        self.experts: torch.Tensor | None = torch.zeros(
            (self.num_experts, self.num_experts)
        )

    def forward(self, x: torch.Tensor):
        router_logits = self.gate(x)

        if isinstance(router_logits, tuple):
            routing_weights = router_logits[0]
            selected_experts = router_logits[-1]
        else:
            routing_weights = router_logits.softmax(dim=1, dtype=torch.float)
            _, selected_experts = torch.topk(routing_weights, TOP_K, dim=-1)
        if self.experts is not None:
            for experts_stack in selected_experts:
                for expert1 in experts_stack:
                    for expert2 in experts_stack:
                        self.experts[expert1, expert2] += 1

        del routing_weights, selected_experts
        torch.cuda.empty_cache()

        return router_logits

    def clear_experts(self):
        self.experts = torch.zeros((self.num_experts, self.num_experts))

    def get_experts(self):
        return self.experts

    def freeze(self):
        self.experts = None


loggers = dict()
for i in LAYERS:
    if isinstance(model.model.layers[i].mlp.gate, ExpertsLogger):
        loggers[i] = ExpertsLogger(model.model.layers[i].mlp.gate.gate)
    else:
        loggers[i] = ExpertsLogger(model.model.layers[i].mlp.gate)
    model.model.layers[i].mlp.gate = loggers[i]


def get_experts(data, loggers):
    with torch.no_grad():
        for logger in loggers.values():
            logger.clear_experts()
        for example in tqdm(data):
            tokenized = tokenizer(example, return_tensors="pt")
            input_ids = tokenized["input_ids"].to(device)
            res = model(input_ids)
            del input_ids, res
            torch.cuda.empty_cache()
        experts = {i: logger.get_experts() for i, logger in loggers.items()}
        for logger in loggers.values():
            logger.freeze()
    return experts


def write_activations(train_data, test_data, loggers):
    act_dir = f"{DATA_DIR}/activations"
    os.makedirs(act_dir, exist_ok=True)
    train_activations = get_experts(train_data, loggers)
    for i, activations in train_activations.items():
        torch.save(activations, f"{act_dir}/train_matrix_{i}.pt")
    test_activations = get_experts(test_data, loggers)
    for i, activations in test_activations.items():
        torch.save(activations, f"{act_dir}/test_matrix_{i}.pt")


write_activations(train_data, test_data, loggers)


# # Extra

# from matplotlib import pyplot as plt
# acts = torch.load(f"{DATA_DIR}/activations/train_matrix_0.pt")
# for i in range(acts.shape[0]):
#     acts[i, i] = 0
# acts /= acts.max()
# plt.imshow(acts)
# plt.show()


# corr = torch.corrcoef(model.model.layers[0].mlp.gate.gate.weight).detach().cpu()


# with torch.no_grad():
#     corr = torch.zeros_like(acts)
#     for _ in tqdm(range(10000)):
#         y = model.model.layers[0].mlp.gate.gate(torch.randn(100, model.model.layers[0].mlp.gate.gate.in_features, device=device))
#         _, selected = torch.topk(y, TOP_K)
#         selected = selected.cpu()
#         for s in selected:
#             for i in s:
#                 for j in s:
#                     corr[i, j] += 1

#     corr = corr.cpu().detach()


# for i in range(corr.shape[0]):
#     corr[i, i] = 0
# corr /= corr.max()
# plt.imshow(corr)
# plt.show()
# plt.imshow((acts - corr) ** 2)
# plt.show()
# print(torch.sum((acts - corr) ** 2))


# def build_matches(corr_matrix):
#     """Get a matrix of correlations and build
#     matching with most sum of correlations (greedy)"""
#     matches = []
#     n = corr_matrix.shape[0]
#     used = torch.zeros(n, dtype=torch.bool)

#     sim_matrix = corr_matrix.clone()

#     sim_matrix.fill_diagonal_(-float("inf"))

#     for _ in range(n // 2):
#         _, max_idx = torch.max(sim_matrix.view(-1), 0)
#         i = max_idx.item() // n
#         j = max_idx.item() % n

#         matches.append((i, j))
#         used[i] = True
#         used[j] = True

#         sim_matrix[i, :] = -float("inf")
#         sim_matrix[:, i] = -float("inf")
#         sim_matrix[j, :] = -float("inf")
#         sim_matrix[:, j] = -float("inf")

#     del sim_matrix, used
#     return matches


# def eval_match(corr_matrix, matches, top_k=8):
#     win = 0
#     for expert1, expert2 in matches:
#         win += corr_matrix[expert1, expert2]
#     return (win / (corr_matrix.sum() / top_k)).item()


# matches = build_matches(acts)
# print("Test matches with greed algorithm:", eval_match(acts, matches))


# matches = build_matches(corr)
# print("Test matches with greed algorithm:", eval_match(acts, matches))


# def main(train_data, test_data, draw=True):
#     with torch.no_grad():
#         res = []
#         best_win = 0
#         best_pair = []
#         for ind in tqdm(range(200)):
#             pairs = (
#                 torch.arange(0, train_data.shape[0])
#                 [torch.randperm(train_data.shape[0])]
#                 .reshape((-1, 2))
#                 .tolist()
#             )
#             pairs = [(x, y) for x, y in pairs]
#             win = eval_match(train_data, pairs)
#             friend = [-1] * train_data.shape[0]
#             for i, j in pairs:
#                 friend[i] = j
#                 friend[j] = i

#             n = train_data.shape[0]

#             def swap(i, j):
#                 if (i, friend[i]) in pairs:
#                     pairs.remove((i, friend[i]))
#                 else:
#                     pairs.remove((friend[i], i))
#                 if (j, friend[j]) in pairs:
#                     pairs.remove((j, friend[j]))
#                 else:
#                     pairs.remove((friend[j], j))
#                 pairs.append((i, j))
#                 pairs.append((friend[i], friend[j]))
#                 fi = friend[i]
#                 fj = friend[j]
#                 friend[fi] = fj
#                 friend[fj] = fi
#                 friend[i] = j
#                 friend[j] = i

#             def unique(data: list):
#                 data.extend([friend[i] for i in data])
#                 return len(set(data)) == len(data)

#             def win_from_swap(i, j):
#                 return (
#                     train_data[i, j] + train_data[friend[i], friend[j]]
#                     - train_data[i, friend[i]] - train_data[j, friend[j]]
#                 ).item()

#             while True:
#                 delta = 0
#                 i, j = -1, -1
#                 for i1 in range(n):
#                     for j1 in range(i1 + 1, n):
#                         new_delta = win_from_swap(i1, j1)
#                         if new_delta > 0:
#                             delta = new_delta
#                             i, j = i1, j1
#                 if delta < 1e-6:
#                     break

#                 swap(i, j)

#             win = eval_match(train_data, pairs)
#             res.append(win)
#             if win > best_win:
#                 best_win = win
#                 best_pair = pairs
#                 print(ind, win)
#         if draw:
#             plt.hist(res)
#             plt.show()
#             plt.plot(res)
#             plt.show()

#     print("Train", eval_match(train_data, best_pair))
#     print("Test", eval_match(test_data, best_pair))
#     return best_pair


# matches = main(acts, acts, False)
# print("Test matches with greed algorithm:", eval_match(acts, matches))


# matches = main(corr, acts, False)
# print("Test matches with greed algorithm:", eval_match(acts, matches))


# torch.cosine_similarity(corr.flatten(), acts.flatten(), dim=0)
