#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import os
from tqdm import tqdm
import transformers
from config import MODEL_NAME, LAYERS, TOP_K


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


# In[2]:


model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
last_layer = max(LAYERS) + 1
while len(model.model.layers) > last_layer:
    del model.model.layers[last_layer]

model.to(device)
torch.cuda.empty_cache()
model


# In[3]:


train_data = {}
test_data = {}
for layer in LAYERS:
    train_data[layer] = torch.load(f"data/mlp_inputs/train_{layer}.pt")
    test_data[layer] = torch.load(f"data/mlp_inputs/test_{layer}.pt")


# In[4]:


def count_activations(data, layer):
    res = []
    for example in tqdm(data):
        _, selected = model.model.layers[layer].mlp.gate(example.to(device)).topk(TOP_K, dim=-1)
        res.append(selected.cpu())
    return res


# In[5]:


os.makedirs("data/activations", exist_ok=True)

for layer in LAYERS:
    data = count_activations(train_data[layer], layer)
    torch.save(data, f"data/activations/train_{layer}.pt")

    data = count_activations(test_data[layer], layer)
    torch.save(data, f"data/activations/test_{layer}.pt")


# In[6]:


train_activations = {}
test_activations = {}
for layer in LAYERS:
    train_activations[layer] = torch.load(f"data/activations/train_{layer}.pt")
    test_activations[layer] = torch.load(f"data/activations/test_{layer}.pt")


# In[7]:


pairs = {}
pairs_ind = {}
for layer in LAYERS:
    pairs[layer] = torch.load(f"data/pairs_{layer}.pt")
    pairs_ind[layer] = [-1] * (2 * len(pairs[layer]))
    for i, (x, y) in enumerate(pairs[layer]):
        pairs_ind[layer][x] = i
        pairs_ind[layer][y] = i


# In[8]:


def get_pairs_activations(activations, layer):
    pairs_activations = []
    for _ in range(len(pairs[layer])):
        pairs_activations.append([])

    for ind, activation in tqdm(enumerate(activations), total=len(activations)):
        for i in range(len(activation)):
            for j in range(i + 1, len(activation)):
                if pairs_ind[layer][activation[i]] == pairs_ind[layer][activation[j]]:
                    pairs_activations[pairs_ind[layer][activation[i]]].append(ind)

    return pairs_activations


# In[9]:


for layer in LAYERS:
    torch.save(get_pairs_activations(train_activations[layer], layer), f"data/activations/pairs_train_{layer}.pt")
    torch.save(get_pairs_activations(test_activations[layer], layer), f"data/activations/pairs_test_{layer}.pt")

