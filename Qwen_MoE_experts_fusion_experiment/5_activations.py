#!/usr/bin/env python
"""Legacy thin wrapper: build activation artifacts via JSON config."""

import torch
import os
from tqdm import tqdm
from config import LAYERS, correct_model_loading, device, DATA_DIR


import sys
sys.path.append("..")
from common.seeding import set_seed
from config import SEED
set_seed(SEED)


if "model" not in globals():
    model, tokenizer = correct_model_loading()

# Remove extra layers
MAX_LAYERS = max(LAYERS) + 1
for i in range(MAX_LAYERS, len(model.model.layers)):
    del model.model.layers[MAX_LAYERS]

torch.cuda.empty_cache()
model = model.to(device)



train_activations = {}
test_activations = {}
for layer in LAYERS:
    train_activations[layer] = torch.load(f"{DATA_DIR}/activations/all/train_all_{layer}.pt")
    test_activations[layer] = torch.load(f"{DATA_DIR}/activations/all/test_all_{layer}.pt")


pairs = {}
pairs_ind = {}
for layer in LAYERS:
    pairs[layer] = torch.load(f"{DATA_DIR}/pairs/{layer}.pt")
    pairs_ind[layer] = [-1] * (2 * len(pairs[layer])) #index of pair for each expert. Len=n_experts
    for i, (x, y) in enumerate(pairs[layer]):
        pairs_ind[layer][x] = i
        pairs_ind[layer][y] = i


def get_pairs_activations(activations, layer):
    pairs_activations = []
    for _ in range(len(pairs[layer])):
        pairs_activations.append(set())

    for ind, activation in tqdm(
        enumerate(activations),
        total=len(activations)
    ):
        topk = len(activation)
        for i in range(topk):
            activated_expert = pairs_ind[layer][activation[i]] #pairs activations (if expert from pair was activated than +1. Index is 'pair')
            pairs_activations[activated_expert].add(ind) #indexes of activations for each pair. If one of experts in pair was in current activation
            #add the index of that activation (n_activation = seq_len)
        # for i in range(topk):
        #     for j in range(i + 1, topk):
        #         if (pairs_ind[layer][activation[i]]
        #                 == pairs_ind[layer][activation[j]]):
        #             activated_expert = pairs_ind[layer][activation[i]]
        #             pairs_activations[activated_expert].append(ind)

    return list(map(list, pairs_activations))


os.makedirs(f"{DATA_DIR}/activations/pairs", exist_ok=True)
for layer in LAYERS:
    torch.save(
        get_pairs_activations(train_activations[layer], layer),
        f"{DATA_DIR}/activations/pairs/train_{layer}.pt"
    )
    torch.save(
        get_pairs_activations(test_activations[layer], layer),
        f"{DATA_DIR}/activations/pairs/test_{layer}.pt"
    )

