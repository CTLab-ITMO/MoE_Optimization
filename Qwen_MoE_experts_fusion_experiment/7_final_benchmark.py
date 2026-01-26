#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
import sys
from config import LAYERS

sys.path.append("..")
from common.fusion_layer import get_fusion_layer_by_index


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[2]:


LAYER = LAYERS[0]


# In[3]:


train_data = torch.load(f"data/mlp_inputs/train_{LAYER}.pt")
test_data = torch.load(f"data/mlp_inputs/test_{LAYER}.pt")


# In[4]:


model = get_fusion_layer_by_index(LAYER, device=device)


# In[5]:


model.forward_default(torch.stack([train_data[0], train_data[1]]).to(device))[0] - model.forward_default(
    torch.stack([torch.stack([train_data[0]]), torch.stack([train_data[1]])]).to(device)
)[0]


# In[6]:


model.forward(torch.stack([train_data[0], train_data[1]]).to(device))[0] - model.forward(
    torch.stack([torch.stack([train_data[0]]), torch.stack([train_data[1]])]).to(device)
)[0]


# In[7]:


batch_size = 1024
total = 100


# In[8]:


for ind in tqdm(range(0, batch_size * total, batch_size)):
    model.forward(torch.stack(train_data[ind : ind + batch_size]).to(device))[0]


# In[9]:


for ind in tqdm(range(0, batch_size * total, batch_size)):
    model.forward_default(torch.stack(train_data[ind : ind + batch_size]).to(device))[0]


# In[10]:


for ind in tqdm(range(0, batch_size * total, batch_size)):
    model.forward(torch.stack(train_data[ind : ind + batch_size]).to(device))[0]
for ind in tqdm(range(0, batch_size * total, batch_size)):
    model.forward_default(torch.stack(train_data[ind : ind + batch_size]).to(device))[0]


# In[11]:


for ind in tqdm(range(0, batch_size * total, batch_size)):
    model.forward(torch.stack(train_data[ind : ind + batch_size]).to(device))[0]


# In[12]:


for ind in tqdm(range(0, batch_size * total, batch_size)):
    model.forward_default(torch.stack(train_data[ind : ind + batch_size]).to(device))[0]


# In[13]:


model.forward(torch.stack(train_data[0:10000]).to(device))[0]


# In[14]:


batch_size = 2048
total = 10


# In[15]:


norm = []
err = []

for ind in tqdm(range(0, batch_size * total, batch_size)):
    x = model.forward(torch.stack(train_data[ind : ind + batch_size]).to(device))[0]
    y = model.forward_default(torch.stack(train_data[ind : ind + batch_size]).to(device))[0]
    norm.append(y.square().mean().cpu().detach())
    err.append((x - y).square().mean().cpu().detach())

norm[0], err[0]


# In[16]:


plt.violinplot([norm, err])
plt.show()
sum(err) / sum(norm)

