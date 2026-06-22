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
    LAYERS,
    TRAIN_SIZE,
    TEST_SIZE,
    device, 
    DATA_DIR
)


import sys
sys.path.append("..")
from common.seeding import set_seed
from config import SEED
set_seed(SEED)


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


if "_train_data" not in globals():
    _train_data = datasets.load_dataset(**TRAIN_DATASET_NAME)["question"]
if "_test_data" not in globals():
    _test_data = datasets.load_dataset(**TEST_DATASET_NAME)["question"]
train_data = _train_data[:TRAIN_SIZE]
test_data = _test_data[:TEST_SIZE]


class Logger(nn.Module):
    """Save all hidden states."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.data = []

    def forward(self, *args, **kwargs):
        for x in args[0].detach().cpu()[0, :, :]:
            self.data.append(x)
        return self.model.forward(*args, **kwargs)

    def get(self):
        res = self.data
        self.data = []
        return res


for n_layer in LAYERS:
    logger = Logger(model.model.layers[n_layer].mlp)
    model.model.layers[n_layer].mlp = logger
    os.makedirs(f"{DATA_DIR}/mlp_inputs", exist_ok=True)

    for data in tqdm(train_data):
        input_ids = tokenizer(data, return_tensors="pt")["input_ids"]
        model.forward(input_ids.to(device))
    torch.save(logger.get(), f"{DATA_DIR}/mlp_inputs/train_{n_layer}.pt")

    for data in tqdm(test_data):
        input_ids = tokenizer(data, return_tensors="pt")["input_ids"]
        model.forward(input_ids.to(device))
    torch.save(logger.get(), f"{DATA_DIR}/mlp_inputs/test_{n_layer}.pt")

    model.model.layers[n_layer].mlp = logger.model

