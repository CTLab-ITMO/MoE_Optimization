import torch
import os
from tqdm import tqdm
import transformers
from common import layout
from experts_fusion.config import MODEL_NAME, MODEL_DTYPE, LAYERS, TOP_K, device, DATA_DIR


from common.seeding import set_seed
from experts_fusion.config import SEED
set_seed(SEED)


if "model" not in globals():
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map=device, dtype=MODEL_DTYPE)
if "tokenizer" not in globals():
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)

# Remove extra layers
MAX_LAYERS = max(LAYERS) + 1
for i in range(MAX_LAYERS, len(model.model.layers)):
    del model.model.layers[MAX_LAYERS]

torch.cuda.empty_cache()
model = model.to(device)


train_data = {}
validation_data = {}
for layer in LAYERS:
    train_data[layer] = torch.load(layout.mlp_inputs_path(DATA_DIR, layout.TRAIN_SPLIT, layer))
    validation_data[layer] = torch.load(layout.mlp_inputs_path(DATA_DIR, layout.VALIDATION_SPLIT, layer))


def count_activations(data, layer):
    """Return sequance of top-k expert on each data example."""
    res = []
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


# Create the directory the stage actually writes to (``activations``); the
# pair-activation and top-k artifacts below all land there.
os.makedirs(layout.activations_dir(DATA_DIR), exist_ok=True)

for layer in LAYERS:
    data = count_activations(train_data[layer], layer)
    torch.save(data, layout.activations_all_path(DATA_DIR, layout.TRAIN_SPLIT, layer))

    data = count_activations(validation_data[layer], layer)
    torch.save(data, layout.activations_all_path(DATA_DIR, layout.VALIDATION_SPLIT, layer))


train_activations = {}
validation_activations = {}
for layer in LAYERS:
    train_activations[layer] = torch.load(layout.activations_all_path(DATA_DIR, layout.TRAIN_SPLIT, layer))
    validation_activations[layer] = torch.load(layout.activations_all_path(DATA_DIR, layout.VALIDATION_SPLIT, layer))


pairs = {}
pairs_ind = {}
for layer in LAYERS:
    pairs[layer] = torch.load(layout.pairs_path(DATA_DIR, layer))
    pairs_ind[layer] = [-1] * (2 * len(pairs[layer]))
    for i, (x, y) in enumerate(pairs[layer]):
        pairs_ind[layer][x] = i
        pairs_ind[layer][y] = i


def get_pairs_activations(activations, layer):
    """Return indexes of all data examples which activate given expert."""
    pairs_activations = []
    for _ in range(len(pairs[layer])):
        pairs_activations.append(set())

    for ind, activation in tqdm(
        enumerate(activations),
        total=len(activations)
    ):
        topk = len(activation)
        for i in range(topk):
            activated_expert = pairs_ind[layer][activation[i]]
            pairs_activations[activated_expert].add(ind)

    return list(map(list, pairs_activations))


for layer in LAYERS:
    torch.save(
        get_pairs_activations(train_activations[layer], layer),
        layout.pair_activations_path(DATA_DIR, layout.TRAIN_SPLIT, layer)
    )
    torch.save(
        get_pairs_activations(validation_activations[layer], layer),
        layout.pair_activations_path(DATA_DIR, layout.VALIDATION_SPLIT, layer)
    )

