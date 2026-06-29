"""Stage 1 -- preprocessing and raw-data capture (§2.2, §3.4, §3.6).

Collapses the old stages 1 (co-activation matrices), 3 (saved experts), 4
(MLP-block inputs) and the top-k half of 5 into one stage with two sub-steps:

* **1a -- weight extraction** (CPU, no dataset): load the model on CPU and save
  every layer's experts (via the model-family ``EXPERT_ADAPTER``), gate and
  shared-expert pathway. This is the dataset-free half and matches the old
  stage 3 exactly (same adapter, same dtype, same files).
* **1b -- capture** (single forward pass per split, non-destructive hooks): load
  the model on ``device`` and register a forward-*pre*-hook on each MoE layer's
  ``mlp``. The hook captures the block input (the MLP-block inputs) and routes it
  through the layer's gate to accumulate the ``EXPERTS x EXPERTS`` co-activation
  matrix and the per-example top-k selections -- all in one pass, instead of the
  three separate model passes the old stages 1/4/5 used. Hooks (not module
  replacement) keep the forward identical to the originals.

The capture is equivalent to the old per-stage passes because the fake oracle's
hidden states are a deterministic function of the input (S8): the captured
artifacts no longer depend on how many forwards ran before them.

The grouping signal (co-activation) does not depend on the grouping itself, so it
lives here in stage 1; the per-group join (top-k x Partition) is stage 2's job
(§3.4 cut line). Finally stage 1 stamps the provenance: ``config_snapshot.json``
(every later stage asserts against it), a ``RunManifest`` and the ``_layout_version``.
"""
import os

import datasets
import torch
import transformers
from tqdm import tqdm

from common import layout, provenance
from common.artifacts import CoActivationMatrix
from common.model_topology import discover_moe_layers
from common.seeding import set_seed
from experts_fusion.config import (
    DATA_DIR,
    DATASET_TEXT_FIELD,
    EXPERT_ADAPTER,
    LAYERS,
    MODEL_DTYPE,
    MODEL_NAME,
    ROUTER_ADAPTER,
    SEED,
    TOP_K,
    TRAIN_DATASET_NAME,
    TRAIN_SIZE,
    VALIDATION_DATASET_NAME,
    VALIDATION_SIZE,
    config_snapshot_dict,
    device,
)

print("Device:", device)

# Stamp the DATA_DIR layout version + the canonical config snapshot first, so a
# re-run against a directory written under a different layout/config fails fast.
layout.write_layout_version(DATA_DIR)
provenance.write_config_snapshot(DATA_DIR, config_snapshot_dict())


# --------------------------------------------------------------------------- #
# 1a -- weight extraction (CPU, no dataset). Matches the old stage 3 exactly.
# --------------------------------------------------------------------------- #
def extract_weights() -> None:
    set_seed(SEED)
    cpu = torch.device("cpu")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map=cpu, dtype=MODEL_DTYPE
    )
    for n_layer in tqdm(discover_moe_layers(model, selected=LAYERS), desc="1a extract"):
        experts = model.model.layers[n_layer].mlp.experts
        os.makedirs(layout.experts_dir(DATA_DIR, n_layer), exist_ok=True)
        for i in range(EXPERT_ADAPTER.count(experts)):
            mlp = EXPERT_ADAPTER.extract(experts, i, device=cpu)
            torch.save(mlp.to(dtype=model.dtype), layout.expert_path(DATA_DIR, n_layer, i))
        torch.save(model.model.layers[n_layer].mlp.gate.cpu(), layout.gate_path(DATA_DIR, n_layer))
        torch.save(
            model.model.layers[n_layer].mlp.shared_expert.cpu(),
            layout.shared_expert_path(DATA_DIR, n_layer),
        )
        torch.save(
            model.model.layers[n_layer].mlp.shared_expert_gate.cpu(),
            layout.shared_expert_gate_path(DATA_DIR, n_layer),
        )


# --------------------------------------------------------------------------- #
# 1b -- capture (single forward pass per split via non-destructive hooks).
# --------------------------------------------------------------------------- #
def _capture_split(model, tokenizer, data, moe_layers, n_experts):
    """Run one forward pass over ``data`` capturing all three raw artifacts.

    Returns ``(coactivation, mlp_inputs, topk)`` keyed by layer:
    ``coactivation[L]`` an ``EXPERTS x EXPERTS`` count matrix (old stage 1),
    ``mlp_inputs[L]`` the list of block-input row vectors (old stage 4), and
    ``topk[L]`` the per-example top-k selection list (old stage 5, first half).
    """
    coactivation = {l: torch.zeros(n_experts, n_experts) for l in moe_layers}
    mlp_inputs = {l: [] for l in moe_layers}
    topk = {l: [] for l in moe_layers}
    handles = []

    def make_hook(layer):
        def hook(module, args):
            # Block input to this layer's mlp. Batch is 1 (one example fed at a
            # time), so [0] selects the sequence rows -- the same vectors the old
            # stage 4 Logger captured and the old stage 5 routed.
            block_rows = args[0].detach()[0, :, :]
            for row in block_rows.cpu():
                mlp_inputs[layer].append(row)
            decision = ROUTER_ADAPTER.route(module.gate(block_rows), top_k=TOP_K)
            for token_experts in decision.selected_experts:
                row_experts = token_experts.flatten()
                topk[layer].append(row_experts.detach().cpu())
                for e1 in row_experts:
                    for e2 in row_experts:
                        coactivation[layer][e1, e2] += 1
            return None

        return hook

    for layer in moe_layers:
        handles.append(
            model.model.layers[layer].mlp.register_forward_pre_hook(make_hook(layer))
        )
    try:
        with torch.no_grad():
            for example in tqdm(data, desc="1b capture"):
                input_ids = tokenizer(example, return_tensors="pt")["input_ids"].to(device)
                model(input_ids)
    finally:
        for handle in handles:
            handle.remove()
    return coactivation, mlp_inputs, topk


def capture(model, tokenizer, n_experts):
    moe_layers = discover_moe_layers(model, selected=LAYERS)
    os.makedirs(layout.activations_dir(DATA_DIR), exist_ok=True)
    os.makedirs(layout.mlp_inputs_dir(DATA_DIR), exist_ok=True)

    train_data = datasets.load_dataset(**TRAIN_DATASET_NAME)[DATASET_TEXT_FIELD][:TRAIN_SIZE]
    validation_data = datasets.load_dataset(**VALIDATION_DATASET_NAME)[DATASET_TEXT_FIELD][:VALIDATION_SIZE]

    for split, data in (
        (layout.TRAIN_SPLIT, train_data),
        (layout.VALIDATION_SPLIT, validation_data),
    ):
        coact, mlp_inputs, topk = _capture_split(model, tokenizer, data, moe_layers, n_experts)
        for layer in moe_layers:
            CoActivationMatrix(coact[layer], split=split, sample_size=len(data)).save(
                layout.activation_matrix_path(DATA_DIR, split, layer)
            )
            torch.save(mlp_inputs[layer], layout.mlp_inputs_path(DATA_DIR, split, layer))
            torch.save(topk[layer], layout.activations_all_path(DATA_DIR, split, layer))
    return validation_data


def main() -> None:
    from experts_fusion.config import EXPERTS

    extract_weights()

    set_seed(SEED)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map=device, dtype=MODEL_DTYPE
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    model = model.to(device)
    validation_data = capture(model, tokenizer, EXPERTS)

    provenance.RunManifest(
        seed=SEED,
        commit=provenance.git_commit(),
        model_id=MODEL_NAME,
        dataset=str(TRAIN_DATASET_NAME),
        dataset_fingerprint=provenance.dataset_fingerprint(
            validation_data, n=VALIDATION_SIZE, seed=SEED
        ),
        config_hash=provenance.config_hash(config_snapshot_dict()),
    ).save(DATA_DIR)


main()
