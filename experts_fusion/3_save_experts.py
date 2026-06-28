import torch
import os
import transformers
from experts_fusion.config import MODEL_NAME, MODEL_DTYPE, LAYERS, DATA_DIR
from tqdm import tqdm

from common import layout
from common.mlp import AdvancedMLP
from common.mlp_utils import turn_in_mlp

device = torch.device("cpu")


from common.seeding import set_seed
from experts_fusion.config import SEED
set_seed(SEED)


if "model" not in globals():
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map=device, dtype=MODEL_DTYPE)


for n_layer in tqdm(LAYERS):
    experts = model.model.layers[n_layer].mlp.experts
    os.makedirs(layout.experts_dir(DATA_DIR, n_layer), exist_ok=True)

    try:
        for i, expert in enumerate(experts):
            torch.save(turn_in_mlp(expert, device=device).to(dtype=model.dtype), layout.expert_path(DATA_DIR, n_layer, i))
    except TypeError:
        for i in range(experts.num_experts):
            new_mlp = AdvancedMLP(
                experts.hidden_dim,
                experts.intermediate_dim,
                experts.hidden_dim,
                device=device,
            )
            new_mlp.gate.weight = torch.nn.Parameter(experts.gate_up_proj[i, :experts.intermediate_dim].detach().clone())
            new_mlp.up.weight = torch.nn.Parameter(experts.gate_up_proj[i, experts.intermediate_dim:].detach().clone())
            new_mlp.down.weight = torch.nn.Parameter(experts.down_proj[i].detach().clone())
            torch.save(new_mlp.to(dtype=model.dtype), layout.expert_path(DATA_DIR, n_layer, i))

    torch.save(
        model.model.layers[n_layer].mlp.gate.cpu(),
        layout.gate_path(DATA_DIR, n_layer),
    )
    torch.save(
        model.model.layers[n_layer].mlp.shared_expert.cpu(),
        layout.shared_expert_path(DATA_DIR, n_layer),
    )
    torch.save(
        model.model.layers[n_layer].mlp.shared_expert_gate.cpu(),
        layout.shared_expert_gate_path(DATA_DIR, n_layer),
    )

