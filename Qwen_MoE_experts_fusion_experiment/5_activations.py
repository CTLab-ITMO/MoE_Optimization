#!/usr/bin/env python
# coding: utf-8

import torch
import os
from tqdm import tqdm
import transformers
from config import MODEL_NAME, LAYERS, TOP_K, device


if "model" not in globals():
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map=device)
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
    train_data[layer] = torch.load(f"data/mlp_inputs/train_{layer}.pt")
    test_data[layer] = torch.load(f"data/mlp_inputs/test_{layer}.pt")


def count_activations(data, layer):
    res = []
    for example in tqdm(data):
        _, selected = (model
                       .model
                       .layers[layer]
                       .mlp
                       .gate(example.to(device))
                       .topk(TOP_K, dim=-1))
        res.append(selected.cpu())
    return res


os.makedirs("data/activations", exist_ok=True)

for layer in LAYERS:
    data = count_activations(train_data[layer], layer)
    torch.save(data, f"data/activations/train_{layer}.pt")

    data = count_activations(test_data[layer], layer)
    torch.save(data, f"data/activations/test_{layer}.pt")


train_activations = {}
test_activations = {}
for layer in LAYERS:
    train_activations[layer] = torch.load(f"data/activations/train_{layer}.pt")
    test_activations[layer] = torch.load(f"data/activations/test_{layer}.pt")


pairs = {}
pairs_ind = {}
for layer in LAYERS:
    pairs[layer] = torch.load(f"data/pairs_{layer}.pt")
    pairs_ind[layer] = [-1] * (2 * len(pairs[layer]))
    for i, (x, y) in enumerate(pairs[layer]):
        pairs_ind[layer][x] = i
        pairs_ind[layer][y] = i


def get_pairs_activations(activations, layer):
    pairs_activations = []
    for _ in range(len(pairs[layer])):
        pairs_activations.append([])

    for ind, activation in tqdm(
        enumerate(activations),
        total=len(activations)
    ):
        for i in range(len(activation)):
            for j in range(i + 1, len(activation)):
                if (pairs_ind[layer][activation[i]]
                        == pairs_ind[layer][activation[j]]):
                    activation = pairs_ind[layer][activation[i]]
                    pairs_activations[activation].append(ind)

    return pairs_activations


for layer in LAYERS:
    torch.save(
        get_pairs_activations(train_activations[layer], layer),
        f"data/activations/pairs_train_{layer}.pt"
    )
    torch.save(
        get_pairs_activations(test_activations[layer], layer),
        f"data/activations/pairs_test_{layer}.pt"
    )

