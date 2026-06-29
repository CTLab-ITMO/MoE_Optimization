import torch
import os
import transformers
from experts_fusion.config import MODEL_NAME, MODEL_DTYPE, LAYERS, EXPERT_ADAPTER, DATA_DIR
from tqdm import tqdm

from common import layout
from common.model_topology import discover_moe_layers

device = torch.device("cpu")


from common.seeding import set_seed
from experts_fusion.config import SEED
set_seed(SEED)


if "model" not in globals():
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map=device, dtype=MODEL_DTYPE)


# Expert extraction goes through the model-family adapter (§3.2): it reproduces
# both the ModuleList layout (shared gate/up/down projections) and the combined
# gate_up_proj/down_proj layout, and raises a clear error on quantized experts.
# discover_moe_layers replaces the blind range/AttributeError-prone access so a
# hybrid dense+sparse stack skips its dense layers (Qwen1.5: every layer is MoE).
for n_layer in tqdm(discover_moe_layers(model, selected=LAYERS)):
    experts = model.model.layers[n_layer].mlp.experts
    os.makedirs(layout.experts_dir(DATA_DIR, n_layer), exist_ok=True)

    for i in range(EXPERT_ADAPTER.count(experts)):
        mlp = EXPERT_ADAPTER.extract(experts, i, device=device)
        torch.save(mlp.to(dtype=model.dtype), layout.expert_path(DATA_DIR, n_layer, i))

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

