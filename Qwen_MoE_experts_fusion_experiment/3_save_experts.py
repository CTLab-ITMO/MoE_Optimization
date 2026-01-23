#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import os
import transformers
from config import MODEL_NAME, LAYERS


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[2]:


model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_NAME)


# In[3]:


for n_layer in LAYERS:
    experts = model.model.layers[n_layer].mlp.experts
    os.makedirs(f"data/experts/{n_layer}", exist_ok=True)
    for i, expert in enumerate(experts):
        torch.save(expert, f"data/experts/{n_layer}/{i}.pt")
    torch.save(model.model.layers[n_layer].mlp.gate, f"data/experts/{n_layer}/gate.pt")
    torch.save(
        model.model.layers[n_layer].mlp.shared_expert,
        f"data/experts/{n_layer}/shared_expert.pt",
    )
    torch.save(
        model.model.layers[n_layer].mlp.shared_expert_gate,
        f"data/experts/{n_layer}/shared_expert_gate.pt",
    )

