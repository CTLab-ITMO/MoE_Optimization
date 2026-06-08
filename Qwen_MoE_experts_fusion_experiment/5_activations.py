#!/usr/bin/env python
# coding: utf-8

import torch
import os
from tqdm import tqdm
import transformers
from config import MODEL_NAME, LAYERS, TOP_K, device, DATA_DIR


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


train_data = {}
test_data = {}
for layer in LAYERS:
    train_data[layer] = torch.load(f"{DATA_DIR}/mlp_inputs/train_{layer}.pt")
    test_data[layer] = torch.load(f"{DATA_DIR}/mlp_inputs/test_{layer}.pt")


def count_activations(data, layer):
    res = []
    for example in tqdm(data):
        router_output = model.model.layers[layer].mlp.gate(example.to(device))
        if isinstance(router_output, tuple):
            selected = router_output[-1]
        else:
            _, selected = router_output.topk(TOP_K, dim=-1)

        res.append(selected.cpu().flatten())
    return res


os.makedirs(f"{DATA_DIR}/mlp_inputs", exist_ok=True)

for layer in LAYERS:
    data = count_activations(train_data[layer], layer)
    torch.save(data, f"{DATA_DIR}/activations/train_all_{layer}.pt")

    data = count_activations(test_data[layer], layer)
    torch.save(data, f"{DATA_DIR}/activations/test_all_{layer}.pt")


train_activations = {}
test_activations = {}
for layer in LAYERS:
    train_activations[layer] = torch.load(
        f"{DATA_DIR}/activations/train_all_{layer}.pt"
    )
    test_activations[layer] = torch.load(f"{DATA_DIR}/activations/test_all_{layer}.pt")


pairs = {}
pairs_ind = {}
for layer in LAYERS:
    pairs[layer] = torch.load(f"{DATA_DIR}/pairs/{layer}.pt")
    pairs_ind[layer] = [-1] * (2 * len(pairs[layer]))
    for i, (x, y) in enumerate(pairs[layer]):
        pairs_ind[layer][x] = i
        pairs_ind[layer][y] = i


def get_pairs_activations(activations, layer):
    pairs_activations = []
    for _ in range(len(pairs[layer])):
        pairs_activations.append(set())

    for ind, activation in tqdm(enumerate(activations), total=len(activations)):
        topk = len(activation)
        for i in range(topk):
            activated_expert = pairs_ind[layer][activation[i]]
            pairs_activations[activated_expert].add(ind)
        # for i in range(topk):
        #     for j in range(i + 1, topk):
        #         if (pairs_ind[layer][activation[i]]
        #                 == pairs_ind[layer][activation[j]]):
        #             activated_expert = pairs_ind[layer][activation[i]]
        #             pairs_activations[activated_expert].append(ind)

    return list(map(list, pairs_activations))


for layer in LAYERS:
    torch.save(
        get_pairs_activations(train_activations[layer], layer),
        f"{DATA_DIR}/activations/pairs_train_{layer}.pt",
    )
    torch.save(
        get_pairs_activations(test_activations[layer], layer),
        f"{DATA_DIR}/activations/pairs_test_{layer}.pt",
    )
