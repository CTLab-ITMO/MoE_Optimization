#!/usr/bin/env python
"""Legacy thin wrapper: find H1 expert pairs from JSON config."""

import sys
import os
from datetime import datetime

import torch
from matplotlib import pyplot as plt

sys.path.append("..")
from common.pairing import (
    get_pairs,
    compute_activation_correlation_matrix,
)
from config import (
    LAYERS,
    EVOLUTION_STEPS,
    EXPERTS,
    # PAIRING_METHOD,
    DATA_DIR,
    correct_model_loading,
    device,
)

PAIRING_METHOD = os.getenv("PAIRING_METHOD")

if "model" not in globals():
    model, _ = correct_model_loading()

print(f"PAIRING_METHOD = {PAIRING_METHOD}")


import sys
sys.path.append("..")
from common.seeding import set_seed
from config import SEED
set_seed(SEED)

if PAIRING_METHOD == "activation_correlation":
    sim0 = compute_activation_correlation_matrix(LAYERS[0], DATA_DIR)
    now = datetime.now()
    _plot_dir = f"plots/activation_correlation/{now.strftime('%Y_%m_%d_%H_%M_%S')}"
    os.makedirs(_plot_dir, exist_ok=True)
    plt.imshow(sim0)
    plt.savefig(f"{_plot_dir}/train_layer{LAYERS[0]}.png")
    plt.close()


evolution_steps = EVOLUTION_STEPS

best_pairs = {}
for layer in LAYERS:
    pairs = get_pairs(
        method=PAIRING_METHOD,
        layer=layer,
        n_experts=EXPERTS,
        data_dir=DATA_DIR,
        device=device,
        evolution_steps=evolution_steps,
        progress=True,
        model=model
    )
    best_pairs[layer] = pairs
    print(f"layer {layer}: {len(pairs)} pairs")


os.makedirs(f"{DATA_DIR}/pairs", exist_ok=True)
for layer in LAYERS:
    torch.save(best_pairs[layer], f"{DATA_DIR}/pairs/{layer}.pt")

print("Saved data example:", torch.load(f"{DATA_DIR}/pairs/{LAYERS[0]}.pt"))

