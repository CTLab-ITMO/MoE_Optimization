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
    device
)

print("Device:", device)


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
print("Example of train data:", train_data[0])
print("Example of test data:", test_data[0])


class ExpertsLogger(nn.Module):
    """Simple class for saving activated experts based on gate"""

    def __init__(self, gate: nn.Module):
        super().__init__()
        self.gate = gate
        self.num_experts = gate.out_features
        self.experts: torch.Tensor | None = torch.zeros(
            (self.num_experts, self.num_experts)
        )

    def forward(self, x: torch.Tensor):
        router_logits = self.gate(x)

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
    os.makedirs("data/activations", exist_ok=True)
    train_activations = get_experts(train_data, loggers)
    for i, activations in train_activations.items():
        torch.save(activations, f"data/activations/train_{i}.pt")
    test_activations = get_experts(test_data, loggers)
    for i, activations in test_activations.items():
        torch.save(activations, f"data/activations/test_{i}.pt")


write_activations(train_data, test_data, loggers)

