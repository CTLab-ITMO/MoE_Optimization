#!/usr/bin/env python
"""Legacy thin wrapper: export experts via JSON config."""

import sys
import torch
import os
import transformers
from config import MODEL_NAME, LAYERS, DATA_DIR
from tqdm import tqdm

sys.path.append("..")
from common.mlp import AdvancedMLP
from common.mlp_utils import turn_in_mlp

device = torch.device("cpu")


import sys
sys.path.append("..")
from common.seeding import set_seed
from config import SEED
set_seed(SEED)


if "model" not in globals():
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map=device)


for n_layer in tqdm(LAYERS):
    experts = model.model.layers[n_layer].mlp.experts
    os.makedirs(f"{DATA_DIR}/experts/{n_layer}", exist_ok=True)

    try:
        for i, expert in enumerate(experts):
            torch.save(turn_in_mlp(expert, device = torch.device("cpu")).to(dtype=model.dtype), f"{DATA_DIR}/experts/{n_layer}/expert_{i}.pt")
    except TypeError:
        for i in range(experts.num_experts):
            new_mlp = AdvancedMLP(
                experts.hidden_dim,
                experts.intermediate_dim,
                experts.hidden_dim,
                device=torch.device("cpu"),
            )
            new_mlp.gate.weight = torch.nn.Parameter(experts.gate_up_proj[i, :experts.intermediate_dim].detach().clone())
            new_mlp.up.weight = torch.nn.Parameter(experts.gate_up_proj[i, experts.intermediate_dim:].detach().clone())
            new_mlp.down.weight = torch.nn.Parameter(experts.down_proj[i].detach().clone())
            torch.save(new_mlp.to(dtype=model.dtype), f"{DATA_DIR}/experts/{n_layer}/expert_{i}.pt")

    torch.save(
        model.model.layers[n_layer].mlp.gate.cpu(),
        f"{DATA_DIR}/experts/{n_layer}/gate.pt"
    )
    torch.save(
        model.model.layers[n_layer].mlp.shared_expert.cpu(),
        f"{DATA_DIR}/experts/{n_layer}/shared_expert.pt",
    )
    torch.save(
        model.model.layers[n_layer].mlp.shared_expert_gate.cpu(),
        f"{DATA_DIR}/experts/{n_layer}/shared_expert_gate.pt",
    )

