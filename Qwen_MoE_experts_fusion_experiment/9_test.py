#!/usr/bin/env python
# coding: utf-8

import os
import sys
import torch
import transformers
from matplotlib import pyplot as plt
from tqdm import tqdm
from config import COMPARE_WITH_ZERO_FUSION_IN_9_SCRIPT, MODEL_NAME, LAYERS, VALIDATION_SIZE, correct_model_loading, REPLACE_LAYERS_SEQUANTIAL, DATA_DIR, REPLACE_BEST_LAYERS_SEQUANTIAL, FILTER_FUSIONS, FILTER_METRICS, REPLACE_LAYERS_SMART_SEQUANTIAL, TRAIN_DATASET_NAME, TEST_DATASET_NAME, is_deepseek_experiment
import gc
import numpy as np

sys.path.append("..")
from common.fusion_layer import get_fusion_layer_by_index
from common.mlp_utils import FUSION_METHODS, best_metrics
from common.qa_eval import evaluate_qa

use_default_experts = False


def _dataset_tag(ds):
    return f"{ds['path'].split('/')[-1]}_{ds['split']}"

PAIRING_METHOD = os.getenv("PAIRING_METHOD")
TEST_RESULTS_DIR = (
    f"{DATA_DIR}/results/{PAIRING_METHOD}/test_result_{_dataset_tag(TRAIN_DATASET_NAME)}"
    f"_{_dataset_tag(TEST_DATASET_NAME)}"
)

if FILTER_FUSIONS is not None:
    FUSION_METHODS = {k:v for k, v in FUSION_METHODS.items() if k in FILTER_FUSIONS}
if FILTER_METRICS is not None:
    best_metrics = {k:v for k, v in best_metrics.items() if k in FILTER_METRICS}


import sys
sys.path.append("..")
from common.seeding import set_seed
from config import SEED
set_seed(SEED)


def free_model(model):
    try:
        model.cpu()
    except Exception:
        pass
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)


base_accuracy = None
base_size = None

def count_params(model):
    return sum(p.numel() for p in model.parameters())


class ZeroLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        return torch.zeros_like(args[0])

def zero_fusion(model, layer):
    return ZeroLayer().to(model.device, dtype=model.dtype)


def default_fusion(name):
    def inner(model, layer):
        new_layer = get_fusion_layer_by_index(layer, f"fusion_experts/{name}", device=model.device,
                                            use_default_experts=use_default_experts, model=model
                                            ).to(model.device, dtype=model.dtype)
        for expert in new_layer.fusion_experts:
            if expert.U is not None:
                expert.U = expert.U.to(model.device, dtype=model.dtype)
        
        if (is_deepseek_experiment()):
            new_layer.forward = new_layer.forward_deepseek
        
        return new_layer

    return inner


experts_fusions = {name:default_fusion(name) for name in FUSION_METHODS.keys()}
if (COMPARE_WITH_ZERO_FUSION_IN_9_SCRIPT):
    experts_fusions["zero"] = zero_fusion
for best_metric in best_metrics.keys():
    experts_fusions[f"Best_{best_metric}"] = default_fusion(f"Best_{best_metric}")


for fusion_name, experts_fusion in experts_fusions.items():
    print(f"Testing {fusion_name}")
    model, _ = correct_model_loading()

    if base_accuracy is None:
        base_accuracy = evaluate_qa(model, tokenizer, TEST_DATASET_NAME, VALIDATION_SIZE, seed=SEED)
        base_size = count_params(model)
    accuracy_list = [base_accuracy]
    size_list = [base_size]

    for layer in tqdm(LAYERS):
        if REPLACE_LAYERS_SEQUANTIAL:
            del model.model.layers[layer].mlp
            torch.cuda.empty_cache()
            model.model.layers[layer].mlp = experts_fusion(model, layer)
            accuracy_list.append(evaluate_qa(model, tokenizer, TEST_DATASET_NAME, VALIDATION_SIZE, seed=SEED))
            size_list.append(count_params(model))
        else:
            old_mlp = model.model.layers[layer].mlp
            model.model.layers[layer].mlp = experts_fusion(model, layer)
            accuracy_list.append(evaluate_qa(model, tokenizer, TEST_DATASET_NAME, VALIDATION_SIZE, seed=SEED))
            size_list.append(count_params(model))
            del model.model.layers[layer].mlp
            torch.cuda.empty_cache()
            model.model.layers[layer].mlp = old_mlp

    mode_str = 'seq' if REPLACE_LAYERS_SEQUANTIAL else 'single'
    print(f"model's accuracy list after {fusion_name} ({mode_str}):", accuracy_list)
    os.makedirs(f"{TEST_RESULTS_DIR}/accuracy", exist_ok=True)
    os.makedirs(f"{TEST_RESULTS_DIR}/model_size", exist_ok=True)
    torch.save(accuracy_list, f"{TEST_RESULTS_DIR}/accuracy/accuracy_list_{mode_str}_{fusion_name}.pt")
    torch.save(size_list, f"{TEST_RESULTS_DIR}/model_size/model_size_list_{mode_str}_{fusion_name}.pt")

    if REPLACE_BEST_LAYERS_SEQUANTIAL:
        best_layers_indexes = [x[0] for x in sorted(enumerate(accuracy_list[1:]), key=lambda x:-x[1])]
        best_layers = np.array(LAYERS)[best_layers_indexes]
        print("Best layers:", best_layers)
        new_accuracy_list = [base_accuracy]
        new_size_list = [base_size]

        for layer in tqdm(best_layers):
            del model.model.layers[layer].mlp
            torch.cuda.empty_cache()
            model.model.layers[layer].mlp = experts_fusion(model, layer)
            new_accuracy_list.append(evaluate_qa(model, tokenizer, TEST_DATASET_NAME, VALIDATION_SIZE, seed=SEED))
            new_size_list.append(count_params(model))
        print(f"model's accuracy list after {fusion_name} (seq):", new_accuracy_list)
        torch.save(new_accuracy_list, f"{TEST_RESULTS_DIR}/accuracy/accuracy_list_seq_{fusion_name}.pt")
        torch.save(new_size_list, f"{TEST_RESULTS_DIR}/model_size/model_size_list_seq_{fusion_name}.pt")
    free_model(model)
    model = None


if REPLACE_LAYERS_SMART_SEQUANTIAL:
    for fusion_name, experts_fusion in tqdm(list(experts_fusions.items())):
        changed = set()
        remain = set(LAYERS)
        print(f"Testing {fusion_name}")
        model, _ = correct_model_loading()
        if base_accuracy is None:
            base_accuracy = evaluate_qa(model, tokenizer, TEST_DATASET_NAME, VALIDATION_SIZE, seed=SEED)
            base_size = count_params(model)
        accuracy_list = [base_accuracy]
        size_list = [base_size]
        for _ in tqdm(LAYERS):
            best_ind = -1
            best_score = -1

            for layer in remain:
                old_mlp = model.model.layers[layer].mlp
                model.model.layers[layer].mlp = experts_fusion(model, layer)
                new_score = evaluate_qa(model, tokenizer, TEST_DATASET_NAME, VALIDATION_SIZE, seed=SEED)
                if new_score > best_score:
                    best_score = new_score
                    best_ind = layer
                del model.model.layers[layer].mlp
                torch.cuda.empty_cache()
                model.model.layers[layer].mlp = old_mlp
            remain.remove(best_ind)
            changed.add(best_ind)
            model.model.layers[best_ind].mlp = experts_fusion(model, best_ind)
            accuracy_list.append(best_score)
            size_list.append(count_params(model))
        print(f"model's accuracy list after {fusion_name} smart seq:", accuracy_list)
        os.makedirs(f"{TEST_RESULTS_DIR}/accuracy", exist_ok=True)
        os.makedirs(f"{TEST_RESULTS_DIR}/model_size", exist_ok=True)
        torch.save(accuracy_list, f"{TEST_RESULTS_DIR}/accuracy/accuracy_list_smart_seq_{fusion_name}.pt")
        torch.save(size_list, f"{TEST_RESULTS_DIR}/model_size/model_size_list_smart_seq_{fusion_name}.pt")
        free_model(model)
        model = None

