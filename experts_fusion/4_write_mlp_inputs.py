import torch
import torch.nn as nn
import os
from tqdm import tqdm
import transformers
import datasets
from common import layout
from common.model_topology import discover_moe_layers
from experts_fusion.config import (
    MODEL_NAME,
    MODEL_DTYPE,
    TRAIN_DATASET_NAME,
    VALIDATION_DATASET_NAME,
    DATASET_TEXT_FIELD,
    LAYERS,
    TRAIN_SIZE,
    VALIDATION_SIZE,
    device,
    DATA_DIR
)


from common.seeding import set_seed
from experts_fusion.config import SEED
set_seed(SEED)


if "model" not in globals():
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map=device, dtype=MODEL_DTYPE)
if "tokenizer" not in globals():
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)

torch.cuda.empty_cache()
model = model.to(device)

# MoE layers to capture inputs for (§3.2): discovery instead of range(N)+del; on
# Qwen1.5-MoE this is exactly LAYERS, while a hybrid stack skips dense layers.
moe_layers = discover_moe_layers(model, selected=LAYERS)


if "_train_data" not in globals():
    _train_data = datasets.load_dataset(**TRAIN_DATASET_NAME)[DATASET_TEXT_FIELD]
if "_validation_data" not in globals():
    _validation_data = datasets.load_dataset(**VALIDATION_DATASET_NAME)[DATASET_TEXT_FIELD]
train_data = _train_data[:TRAIN_SIZE]
validation_data = _validation_data[:VALIDATION_SIZE]


class Logger(nn.Module):
    """Save all hidden states."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.data = []

    def forward(self, *args, **kwargs):
        for x in args[0].detach().cpu()[0, :, :]:
            self.data.append(x)
        return self.model.forward(*args, **kwargs)

    def get(self):
        res = self.data
        self.data = []
        return res


for n_layer in moe_layers:
    logger = Logger(model.model.layers[n_layer].mlp)
    model.model.layers[n_layer].mlp = logger
    os.makedirs(layout.mlp_inputs_dir(DATA_DIR), exist_ok=True)

    for data in tqdm(train_data):
        input_ids = tokenizer(data, return_tensors="pt")["input_ids"]
        model.forward(input_ids.to(device))
    torch.save(logger.get(), layout.mlp_inputs_path(DATA_DIR, layout.TRAIN_SPLIT, n_layer))

    for data in tqdm(validation_data):
        input_ids = tokenizer(data, return_tensors="pt")["input_ids"]
        model.forward(input_ids.to(device))
    torch.save(logger.get(), layout.mlp_inputs_path(DATA_DIR, layout.VALIDATION_SPLIT, n_layer))

    model.model.layers[n_layer].mlp = logger.model

