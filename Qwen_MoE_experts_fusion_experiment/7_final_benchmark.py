#!/usr/bin/env python
# coding: utf-8

import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
import sys
from config import LAYERS, device, DATA_DIR
import time
import seaborn as sns
from datetime import datetime
import os

sys.path.append("..")
from common.fusion_layer import get_fusion_layer_by_index


import sys
sys.path.append("..")
from common.seeding import set_seed
from config import SEED
set_seed(SEED)


LAYER = LAYERS[0]


train_data = torch.load(f"{DATA_DIR}/mlp_inputs/train_{LAYER}.pt")
validation_data = torch.load(f"{DATA_DIR}/mlp_inputs/validation_{LAYER}.pt")


model = get_fusion_layer_by_index(LAYER, f"fusion_experts/ZipIt", DATA_DIR, device=device)


batch_size = 128
total = 100
total = min(total, len(train_data) // batch_size)


with torch.no_grad():
    with torch.autocast(device.type):
        for ind in tqdm(range(0, batch_size * total, batch_size)):
            batch = torch.stack(train_data[ind: ind + batch_size]).unsqueeze(0)
            batch = batch.to(device)
            model.forward(batch)[0]


with torch.no_grad():
    with torch.autocast(device.type):
        for ind in tqdm(range(0, batch_size * total, batch_size)):
            batch = torch.stack(train_data[ind: ind + batch_size]).unsqueeze(0)
            batch = batch.to(device)
            model.forward_default(batch)[0]


norm = []
err = []
cos_sim = []

time_new = []
time_base = []

with torch.no_grad():
    with torch.autocast(device.type):
        for ind in tqdm(range(0, batch_size * total, batch_size)):
            input = torch.stack(train_data[ind: ind + batch_size]).to(device).unsqueeze(0)
            time_start = time.perf_counter()
            x = model.forward(input)[0]
            time_middle = time.perf_counter()
            y = model.forward_default(input)[0]
            time_end = time.perf_counter()
            for i in range(y.shape[0]):
                norm.append(y[i].square().sum().sqrt().cpu().detach().item())
                err.append((x[i] - y[i]).square().sum().sqrt().cpu().detach().item())
                cos_sim.append(torch
                            .cosine_similarity(
                                x[i].flatten(),
                                y[i].flatten(),
                                dim=0)
                            .cpu()
                            .detach()
                            .item())
            time_new.append(time_middle-time_start)
            time_base.append(time_end-time_middle)

print("Norm:", norm[0])
print("Error:", err[0])
print("Cos_sim:", cos_sim[0])


now = datetime.now()
time = now.strftime("%Y_%m_%d_%H_%M_%S")
_plot_dir = f"plots/fusion/{time}"
os.makedirs(_plot_dir, exist_ok=True)


sns.kdeplot(cos_sim, label='Cosine similarity',
            fill=True, alpha=0.3, log_scale=True)
plt.savefig(f"{_plot_dir}/kdeplot.png")
plt.close()

sns.kdeplot([1e-10 + x for x in norm],
            label='Norm of the true vector',
            fill=True, alpha=0.3, log_scale=True)
sns.kdeplot([1e-10 + x for x in err if x != 0],
            label='Norm of the error vector',
            fill=True, alpha=0.3, log_scale=True)
plt.xlabel('Norm')
plt.ylabel('Density')
plt.title('Density of norms')
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig(f"{_plot_dir}/kdeplot_norms.png")
plt.close()

sns.kdeplot(time_new, label='Our method', fill=True, alpha=0.3)
sns.kdeplot(time_base, label='Original model', fill=True, alpha=0.3)
plt.xlabel('Time')
plt.ylabel('Density')
plt.title('Density of time')
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig(f"{_plot_dir}/kdeplot_times.png")
plt.close()

