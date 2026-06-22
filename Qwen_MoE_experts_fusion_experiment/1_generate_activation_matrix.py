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
    VALIDATION_DATASET_NAME,
    TRAIN_SIZE,
    VALIDATION_SIZE,
    LAYERS,
    TOP_K,
    device,
    DATA_DIR
)

print("Device:", device)


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
if "_validation_data" not in globals():
    _validation_data = datasets.load_dataset(**VALIDATION_DATASET_NAME)["question"]
train_data = _train_data[:TRAIN_SIZE]
validation_data = _validation_data[:VALIDATION_SIZE]
print("Example of train data:", train_data[0])
print("Example of validation data:", validation_data[0])


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


def write_activations(train_data, validation_data, loggers):
    act_dir = f"{DATA_DIR}/activations"
    os.makedirs(act_dir, exist_ok=True)
    train_activations = get_experts(train_data, loggers)
    for i, activations in train_activations.items():
        torch.save(activations, f"{act_dir}/train_matrix_{i}.pt")
    validation_activations = get_experts(validation_data, loggers)
    for i, activations in validation_activations.items():
        torch.save(activations, f"{act_dir}/validation_matrix_{i}.pt")


write_activations(train_data, validation_data, loggers)
