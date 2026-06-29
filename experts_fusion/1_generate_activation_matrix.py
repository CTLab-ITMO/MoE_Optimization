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
    TRAIN_SIZE,
    VALIDATION_SIZE,
    LAYERS,
    TOP_K,
    ROUTER_ADAPTER,
    device,
    DATA_DIR
)

print("Device:", device)


from common.seeding import set_seed
from experts_fusion.config import SEED
set_seed(SEED)


model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map=device, dtype=MODEL_DTYPE)
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)

torch.cuda.empty_cache()
model = model.to(device)


train_data = datasets.load_dataset(**TRAIN_DATASET_NAME)[DATASET_TEXT_FIELD][:TRAIN_SIZE]
validation_data = datasets.load_dataset(**VALIDATION_DATASET_NAME)[DATASET_TEXT_FIELD][:VALIDATION_SIZE]
print("Example of train data:", train_data[0])
print("Example of validation data:", validation_data[0])


class ExpertsLogger(nn.Module):
    """Simple class for saving activated experts based on gate"""

    def __init__(self, gate: nn.Module, router_adapter=ROUTER_ADAPTER):
        super().__init__()
        self.gate = gate
        self.router = router_adapter
        self.num_experts = gate.num_experts
        self.experts: torch.Tensor | None = torch.zeros(
            (self.num_experts, self.num_experts)
        )

    def forward(self, x: torch.Tensor):
        router_logits = self.gate(x)

        # Canonicalize the gate output through the shared router adapter (§3.2):
        # the same adapter decodes the gate at inference (FusionLayer), so the
        # co-activation signal cannot diverge from the real routing. Only the
        # selected experts matter for co-activation counting.
        selected_experts = self.router.route(router_logits, top_k=TOP_K).selected_experts
        if self.experts is not None:
            for experts_stack in selected_experts:
                for expert1 in experts_stack:
                    for expert2 in experts_stack:
                        self.experts[expert1, expert2] += 1

        del selected_experts
        torch.cuda.empty_cache()

        return router_logits

    def clear_experts(self):
        self.experts = torch.zeros((self.num_experts, self.num_experts))

    def get_experts(self):
        return self.experts

    def freeze(self):
        self.experts = None


# Discover which of the requested LAYERS are actually MoE (§3.2) instead of
# trimming the stack with range(N)+del and indexing blindly; on Qwen1.5-MoE every
# layer is MoE, so this yields exactly LAYERS (behaviour-neutral), while a hybrid
# dense+sparse stack skips its dense layers rather than crashing.
loggers = dict()
for i in discover_moe_layers(model, selected=LAYERS):
    if isinstance(model.model.layers[i].mlp.gate, ExpertsLogger):
        loggers[i] = ExpertsLogger(model.model.layers[i].mlp.gate.gate)
    else:
        loggers[i] = ExpertsLogger(model.model.layers[i].mlp.gate)
    model.model.layers[i].mlp.gate = loggers[i]


def get_experts(data, loggers):
    with torch.no_grad():
        for logger in loggers.values():
            logger.clear_experts()
        for example in tqdm(data):
            tokenized = tokenizer(example, return_tensors="pt")
            input_ids = tokenized["input_ids"].to(device)
            res = model(input_ids)
            del input_ids, res
            torch.cuda.empty_cache()
        experts = {i: logger.get_experts() for i, logger in loggers.items()}
        for logger in loggers.values():
            logger.freeze()
    return experts


def write_activations(train_data, validation_data, loggers):
    os.makedirs(layout.activations_dir(DATA_DIR), exist_ok=True)
    train_activations = get_experts(train_data, loggers)
    for i, activations in train_activations.items():
        torch.save(activations, layout.activation_matrix_path(DATA_DIR, layout.TRAIN_SPLIT, i))
    validation_activations = get_experts(validation_data, loggers)
    for i, activations in validation_activations.items():
        torch.save(activations, layout.activation_matrix_path(DATA_DIR, layout.VALIDATION_SPLIT, i))


write_activations(train_data, validation_data, loggers)
