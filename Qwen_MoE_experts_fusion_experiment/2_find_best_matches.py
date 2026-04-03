#!/usr/bin/env python
# coding: utf-8

import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from config import LAYERS, EVOLUTION_STEPS, DATA_DIR
import os
from datetime import datetime


train_data = {}
test_data = {}
for layer in LAYERS:
    train_data[layer] = torch.load(f"{DATA_DIR}/activations/train_matrix_{layer}.pt").cpu()
    test_data[layer] = torch.load(f"{DATA_DIR}/activations/test_matrix_{layer}.pt").cpu()
    train_data[layer] /= train_data[layer].sum()


def build_matches(corr_matrix):
    """Get a matrix of correlations and build 
    matching with most sum of correlations (greedy)"""
    matches = []
    n = corr_matrix.shape[0]
    used = torch.zeros(n, dtype=torch.bool)

    sim_matrix = corr_matrix.clone()

    sim_matrix.fill_diagonal_(-float("inf"))

    for _ in range(n // 2):
        _, max_idx = torch.max(sim_matrix.view(-1), 0)
        i = max_idx.item() // n
        j = max_idx.item() % n

        matches.append((i, j))
        used[i] = True
        used[j] = True

        sim_matrix[i, :] = -float("inf")
        sim_matrix[:, i] = -float("inf")
        sim_matrix[j, :] = -float("inf")
        sim_matrix[:, j] = -float("inf")

    del sim_matrix, used
    return matches


def eval_match(corr_matrix, matches, top_k=8):
    win = 0
    for expert1, expert2 in matches:
        win += corr_matrix[expert1, expert2]
    return (win / (corr_matrix.sum() / top_k)).item()


matches = build_matches(train_data[LAYERS[0]])


print("Test matches with greed algorithm:", eval_match(test_data[LAYERS[0]], matches))


now = datetime.now()
time = now.strftime("%Y_%m_%d_%H_%M_%S")
_plot_dir = f"plots/activation_correlation/{time}"
os.makedirs(_plot_dir, exist_ok=True)

plt.imshow(train_data[LAYERS[0]])
plt.savefig(f"{_plot_dir}/1.png")
plt.close()


plt.imshow(test_data[LAYERS[0]])
plt.savefig(f"{_plot_dir}/2.png")
plt.close()


def main(train_data, test_data, draw=True):
    with torch.no_grad():
        res = []
        best_win = 0
        best_pair = []
        for ind in tqdm(range(EVOLUTION_STEPS)):
            pairs = (
                torch.arange(0, train_data.shape[0])
                [torch.randperm(train_data.shape[0])]
                .reshape((-1, 2))
                .tolist()
            )
            pairs = set([(x, y) for x, y in pairs])
            win = eval_match(train_data, pairs)
            friend = [-1] * train_data.shape[0]
            for i, j in pairs:
                friend[i] = j
                friend[j] = i

            n = train_data.shape[0]

            def swap(i, j):
                if (i, friend[i]) in pairs:
                    pairs.remove((i, friend[i]))
                else:
                    pairs.remove((friend[i], i))
                if (j, friend[j]) in pairs:
                    pairs.remove((j, friend[j]))
                else:
                    pairs.remove((friend[j], j))
                pairs.add((i, j))
                pairs.add((friend[i], friend[j]))
                fi = friend[i]
                fj = friend[j]
                friend[fi] = fj
                friend[fj] = fi
                friend[i] = j
                friend[j] = i

            def unique(data: list):
                data.extend([friend[i] for i in data])
                return len(set(data)) == len(data)

            def win_from_swap(i, j):
                return (
                    train_data[i, j] + train_data[friend[i], friend[j]]
                    - train_data[i, friend[i]] - train_data[j, friend[j]]
                ).item()

            while True:
                delta = 0
                i, j = -1, -1
                for i1 in range(n):
                    for j1 in range(i1 + 1, n):
                        new_delta = win_from_swap(i1, j1)
                        if new_delta > 0:
                            delta = new_delta
                            i, j = i1, j1
                if delta < 1e-6:
                    break

                swap(i, j)

            win = eval_match(train_data, pairs)
            res.append(win)
            if win > best_win:
                best_win = win
                best_pair = pairs
                print(ind, win)
        if draw:
            plt.hist(res)
            plt.show()
            plt.plot(res)
            plt.show()

    best_pair = list(best_pair)
    print("Train", eval_match(train_data, best_pair))
    print("Test", eval_match(test_data, best_pair))
    return best_pair


best_pairs = {
    layer: main(train_data[layer], test_data[layer], draw=False)
    for layer in LAYERS
}


for layer in LAYERS:
    eval_match(train_data[layer], best_pairs[layer])
    eval_match(test_data[layer], best_pairs[layer])


os.makedirs(f"{DATA_DIR}/pairs", exist_ok=True)
for layer in LAYERS:
    torch.save(best_pairs[layer], f"{DATA_DIR}/pairs/{layer}.pt")


print("Saved data example:", torch.load(f"{DATA_DIR}/pairs/{LAYERS[0]}.pt"))

