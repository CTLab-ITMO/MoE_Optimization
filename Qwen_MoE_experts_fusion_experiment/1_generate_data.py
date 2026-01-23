#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import os
from tqdm import tqdm
import transformers
import datasets
from config import MODEL_NAME, DATASET_NAMES, TRAIN_SIZE, TEST_SIZE, LAYERS, TOP_K

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[2]:


if "model" not in globals():
    model = transformers.AutoModel.from_pretrained(MODEL_NAME)
if "tokenizer" not in globals():
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)

# Remove extra layers
MAX_LAYERS = max(LAYERS) + 1
for i in range(MAX_LAYERS, len(model.layers)):
    del model.layers[MAX_LAYERS]

torch.cuda.empty_cache()
model = model.to(device)

if "_train_data" not in globals():
    _train_data = datasets.load_dataset(*DATASET_NAMES, split="train")["question"]
if "_test_data" not in globals():
    _test_data = datasets.load_dataset(*DATASET_NAMES, split="test")["question"]
train_data = _train_data[:TRAIN_SIZE]
test_data = _test_data[:TEST_SIZE]


# In[3]:


class ExpertsLogger(nn.Module):
    """Simple class for saving activated experts"""

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
                        # print(expert1, expert2)
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


# In[4]:


loggers = []
for i in LAYERS:
    if isinstance(model.layers[i].mlp.gate, nn.Linear):
        loggers.append(ExpertsLogger(model.layers[i].mlp.gate))
    else:
        loggers.append(ExpertsLogger(model.layers[i].mlp.gate.gate))
    model.layers[i].mlp.gate = loggers[i]


# In[5]:


def get_experts(data, loggers):
    with torch.no_grad():
        for logger in loggers:
            logger.clear_experts()
        for example in tqdm(data):
            input_ids = tokenizer(example, return_tensors="pt")["input_ids"].to(device)
            res = model(input_ids)
            del input_ids, res
            torch.cuda.empty_cache()
        experts = [logger.get_experts() for logger in loggers]
        for logger in loggers:
            logger.freeze()
    return experts


# In[6]:


def write_activations(train_data, test_data, loggers):
    os.makedirs("data/activations", exist_ok=True)
    train_activations = get_experts(train_data, loggers)
    for i, activations in enumerate(train_activations):
        torch.save(activations, f"data/activations/train_{i}.pt")
    test_activations = get_experts(test_data, loggers)
    for i, activations in enumerate(test_activations):
        torch.save(activations, f"data/activations/test_{i}.pt")


# In[7]:


write_activations(train_data, test_data, loggers)


# In[8]:


del train_data, test_data, loggers
torch.cuda.empty_cache()

