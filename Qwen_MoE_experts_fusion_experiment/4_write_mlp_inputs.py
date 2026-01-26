#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import os
from tqdm import tqdm
import transformers
import datasets
from config import MODEL_NAME, TRAIN_DATASET_NAME, TEST_DATASET_NAME, LAYERS, TRAIN_SIZE, TEST_SIZE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


# In[2]:


model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
last_layer = max(LAYERS) + 1
while len(model.model.layers) > last_layer:
    del model.model.layers[last_layer]
model.model.layers

model.to(device)
torch.cuda.empty_cache()
model.model.layers


# In[3]:


if "_train_data" not in globals():
    _train_data = datasets.load_dataset(*TRAIN_DATASET_NAME, split="train")["question"]
if "_test_data" not in globals():
    _test_data = datasets.load_dataset(*TEST_DATASET_NAME, split="test")["question"]
train_data = _train_data[:TRAIN_SIZE]
test_data = _test_data[:TEST_SIZE]


# In[4]:


class Logger(nn.Module):
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


# In[5]:


for n_layer in LAYERS:
    logger = Logger(model.model.layers[n_layer].mlp)
    model.model.layers[n_layer].mlp = logger
    os.makedirs("data/mlp_inputs", exist_ok=True)

    for data in tqdm(train_data):
        model.forward(tokenizer(data, return_tensors="pt")["input_ids"].to(device))
    torch.save(logger.get(), f"data/mlp_inputs/train_{n_layer}.pt")

    for data in tqdm(test_data):
        model.forward(tokenizer(data, return_tensors="pt")["input_ids"].to(device))
    torch.save(logger.get(), f"data/mlp_inputs/test_{n_layer}.pt")

    model.model.layers[n_layer].mlp = logger.model

