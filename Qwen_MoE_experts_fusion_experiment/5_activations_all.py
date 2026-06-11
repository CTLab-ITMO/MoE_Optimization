import torch
import os
from tqdm import tqdm
from config import LAYERS, TOP_K, correct_model_loading, device, DATA_DIR, is_deepseek_experiment


import sys
sys.path.append("..")
from common.seeding import set_seed
from config import SEED
set_seed(SEED)


if "model" not in globals():
    model, tokenizer = correct_model_loading()

# Remove extra layers
MAX_LAYERS = max(LAYERS) + 1
for i in range(MAX_LAYERS, len(model.model.layers)):
    del model.model.layers[MAX_LAYERS]

torch.cuda.empty_cache()
model = model.to(device)


train_data = {}
test_data = {}
for layer in LAYERS:
    train_data[layer] = torch.load(f"{DATA_DIR}/mlp_inputs/train_{layer}.pt")
    test_data[layer] = torch.load(f"{DATA_DIR}/mlp_inputs/test_{layer}.pt")


def count_activations(data, layer):
    res = []
    if (is_deepseek_experiment()):
        for example in tqdm(data):
            router_output = (model
                            .model
                            .layers[layer]
                            .mlp
                            .gate(example.unsqueeze(0).unsqueeze(0).to(device)))
            if isinstance(router_output, tuple):
                selected = router_output[0]
            else:
                _, selected = router_output.topk(TOP_K, dim=-1)

            res.append(selected.cpu().flatten())
    else:
        for example in tqdm(data):
            router_output = (model
                            .model
                            .layers[layer]
                            .mlp
                            .gate(example.to(device)))
            if isinstance(router_output, tuple):
                selected = router_output[-1]
            else:
                _, selected = router_output.topk(TOP_K, dim=-1)

            res.append(selected.cpu().flatten())
    return res


#We assume that model is one, and experiments are many, so we don't calc activations on the same
#data on the same layers and experts more than once
if (not os.path.exists(f"{DATA_DIR}/activations/all/train_all_{0}.pt")):
    os.makedirs(f"{DATA_DIR}/activations/all/", exist_ok=True)
    for layer in LAYERS:
        data = count_activations(train_data[layer], layer)
        torch.save(data, f"{DATA_DIR}/activations/all/train_all_{layer}.pt")

        data = count_activations(test_data[layer], layer)
        torch.save(data, f"{DATA_DIR}/activations/all/test_all_{layer}.pt")