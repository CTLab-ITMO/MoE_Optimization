"""Stage 1 -- preprocessing and raw-data capture.

Sub-step 1a (CPU, no dataset): save each layer's experts (via EXPERT_ADAPTER),
gate, and shared-expert weights.

Sub-step 1b (single forward pass per split, non-destructive hooks): a
forward-pre-hook on each MoE layer captures, in one pass, the EXPERTS×EXPERTS
co-activation matrix, block-input vectors, and per-example top-k selections.

Also writes provenance: ``config_snapshot.json`` (later stages assert against
it), a ``RunManifest``, and the ``_layout_version`` marker.
"""
import os

import datasets
import torch
import transformers
from tqdm import tqdm

from common import layout, provenance
from common.artifacts import CoActivationMatrix, SavedExpert
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
    SHARED_EXPERT_ADAPTER,
    TOP_K,
    TRAIN_DATASET_NAME,
    TRAIN_SIZE,
    VALIDATION_DATASET_NAME,
    VALIDATION_SIZE,
    config_snapshot_dict,
    device,
)

print("Device:", device)

# Write layout version and config snapshot so a re-run against a mismatched
# directory fails fast.
layout.write_layout_version(DATA_DIR)
provenance.write_config_snapshot(DATA_DIR, config_snapshot_dict())


# --------------------------------------------------------------------------- #
# 1a -- weight extraction (CPU, no dataset).
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
            SavedExpert(mlp.to(dtype=model.dtype)).save(layout.expert_path(DATA_DIR, n_layer, i))
        torch.save(model.model.layers[n_layer].mlp.gate.cpu(), layout.gate_path(DATA_DIR, n_layer))
        # SharedExpertAdapter.extract returns the (shared_expert, shared_expert_gate)
        # submodules, or None for families without a shared expert (e.g. Mixtral) --
        # in which case nothing is written and stage 4's loader falls back to zeros.
        extracted = SHARED_EXPERT_ADAPTER.extract(model.model.layers[n_layer].mlp)
        if extracted is not None:
            shared_expert, shared_expert_gate = extracted
            torch.save(shared_expert.cpu(), layout.shared_expert_path(DATA_DIR, n_layer))
            torch.save(shared_expert_gate.cpu(), layout.shared_expert_gate_path(DATA_DIR, n_layer))


# --------------------------------------------------------------------------- #
# 1b -- capture (single forward pass per split via non-destructive hooks).
# --------------------------------------------------------------------------- #
def _capture_split(model, tokenizer, data, moe_layers, n_experts):
    """Run one forward pass over ``data``, returning three dicts keyed by layer:
    ``coactivation[L]`` (EXPERTS×EXPERTS count matrix), ``mlp_inputs[L]``
    (block-input row vectors), ``topk[L]`` (per-example top-k selections).
    """
    coactivation = {l: torch.zeros(n_experts, n_experts) for l in moe_layers}
    mlp_inputs = {l: [] for l in moe_layers}
    topk = {l: [] for l in moe_layers}
    handles = []

    def make_hook(layer):
        def hook(module, args):
            # [0] selects sequence rows from the batch-of-1 input; saved as
            # MLP inputs and routed through the gate.
            block_rows = args[0].detach()[0, :, :]
            for row in block_rows.cpu():
                mlp_inputs[layer].append(row)
            # route_module owns the gate call, so this hook stays agnostic to how
            # a family's router is invoked (default: module.gate(x) -> decode).
            decision = ROUTER_ADAPTER.route_module(module, block_rows, top_k=TOP_K)
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


def capture(model, tokenizer, n_experts, moe_layers):
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
    # Discover the real MoE layers once (dense layers are dropped from LAYERS on
    # mixed models); capture uses them and the manifest records them so stages 2/3
    # iterate exactly what stage 1 produced (CR#4).
    moe_layers = discover_moe_layers(model, selected=LAYERS)
    validation_data = capture(model, tokenizer, EXPERTS, moe_layers)

    provenance.RunManifest(
        seed=SEED,
        commit=provenance.git_commit(),
        model_id=MODEL_NAME,
        dataset=str(TRAIN_DATASET_NAME),
        dataset_fingerprint=provenance.dataset_fingerprint(
            validation_data, n=VALIDATION_SIZE, seed=SEED
        ),
        config_hash=provenance.config_hash(config_snapshot_dict()),
        produced_layers=moe_layers,
    ).save(DATA_DIR)


main()
