import os
import torch
import transformers
from tqdm import tqdm
from experts_fusion.config import (
    MODEL_NAME,
    MODEL_DTYPE,
    LAYERS,
    EXPERTS,
    TRAIN_SIZE,
    TEST_SIZE,
    device,
    REPLACE_LAYERS_SEQUANTIAL,
    REPLACE_BEST_LAYERS_SEQUANTIAL,
    FILTER_FUSIONS,
    FILTER_METRICS,
    REPLACE_LAYERS_SMART_SEQUANTIAL,
    TRAIN_DATASET_NAME,
    TEST_DATASET_NAME,
    DATA_DIR,
    USE_DEFAULT_EXPERTS,
)
import gc

from common import layout
from common.fusion import FUSION_REGISTRY
from common.fusion_layer import get_fusion_layer_by_index
from common.metrics import METRIC_REGISTRY
from common.mlp_utils import FUSION_METHODS, best_metrics
from common.names import BEST_PREFIX, TEST_RESULT_PREFIX, ZERO_FUSION, ReplaceMode
from common.qa_eval import evaluate_qa
from common.registry import require_all_registered

use_default_experts = USE_DEFAULT_EXPERTS


def _dataset_tag(ds):
    return f"{ds['path'].split('/')[-1]}_{ds['split']}"


TEST_RESULTS_DIR = (
    f"{TEST_RESULT_PREFIX}{_dataset_tag(TRAIN_DATASET_NAME)}"
    f"_{_dataset_tag(TEST_DATASET_NAME)}"
)

# Validate config filters against the registries before applying them (§1.6).
require_all_registered(FILTER_FUSIONS, FUSION_REGISTRY, field="FILTER_FUSIONS")
require_all_registered(FILTER_METRICS, METRIC_REGISTRY, field="FILTER_METRICS")

if FILTER_FUSIONS is not None:
    FUSION_METHODS = {k:v for k, v in FUSION_METHODS.items() if k in FILTER_FUSIONS}
if FILTER_METRICS is not None:
    best_metrics = {k:v for k, v in best_metrics.items() if k in FILTER_METRICS}


from common.seeding import set_seed
from experts_fusion.config import SEED
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


base_train_accuracy = None
base_test_accuracy = None
base_size = None

def count_params(model):
    return sum(p.numel() for p in model.parameters())


class ZeroLayer(torch.nn.Module):
    """Simulate case with zero output of MLP layer."""
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        return torch.zeros_like(args[0])

def zero_fusion(model, layer):
    return ZeroLayer().to(model.device, dtype=model.dtype)


def default_fusion(name):
    def inner(model, layer):
        new_layer = get_fusion_layer_by_index(layer,
                                            layout.fusion_method_dir(name),
                                            DATA_DIR,
                                            device=model.device,
                                            use_default_experts=use_default_experts,
                                            num_experts=EXPERTS
                                            ).to(model.device, dtype=model.dtype)
        for expert in new_layer.fusion_experts:
            if expert.U is not None:
                expert.U = expert.U.to(model.device, dtype=model.dtype)
        return new_layer

    return inner


experts_fusions = {name:default_fusion(name) for name in FUSION_METHODS.keys()}
experts_fusions[ZERO_FUSION] = zero_fusion
for best_metric in best_metrics.keys():
    experts_fusions[f"{BEST_PREFIX}{best_metric}"] = default_fusion(f"{BEST_PREFIX}{best_metric}")


for fusion_name, experts_fusion in experts_fusions.items():
    print(f"Testing {fusion_name}")
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map=device, dtype=MODEL_DTYPE)

    if base_train_accuracy is None:
        base_train_accuracy = evaluate_qa(model, tokenizer, TRAIN_DATASET_NAME, TRAIN_SIZE, seed=SEED)
        base_test_accuracy = evaluate_qa(model, tokenizer, TEST_DATASET_NAME, TEST_SIZE, seed=SEED)
        base_size = count_params(model)
    accuracy_list = [base_train_accuracy]
    size_list = [base_size]

    for layer in tqdm(LAYERS):
        if REPLACE_LAYERS_SEQUANTIAL:
            del model.model.layers[layer].mlp
            torch.cuda.empty_cache()
            model.model.layers[layer].mlp = experts_fusion(model, layer)
            accuracy_list.append(evaluate_qa(model, tokenizer, TRAIN_DATASET_NAME, TRAIN_SIZE, seed=SEED))
            size_list.append(count_params(model))
        else:
            old_mlp = model.model.layers[layer].mlp
            model.model.layers[layer].mlp = experts_fusion(model, layer)
            accuracy_list.append(evaluate_qa(model, tokenizer, TRAIN_DATASET_NAME, TRAIN_SIZE, seed=SEED))
            size_list.append(count_params(model))
            del model.model.layers[layer].mlp
            torch.cuda.empty_cache()
            model.model.layers[layer].mlp = old_mlp

    mode_str = ReplaceMode.SEQ if REPLACE_LAYERS_SEQUANTIAL else ReplaceMode.SINGLE
    print(f"model's accuracy list after {fusion_name} ({mode_str}):", accuracy_list)
    os.makedirs(f"{TEST_RESULTS_DIR}/accuracy", exist_ok=True)
    os.makedirs(f"{TEST_RESULTS_DIR}/model_size", exist_ok=True)
    torch.save(accuracy_list, f"{TEST_RESULTS_DIR}/accuracy/accuracy_list_{mode_str}_{fusion_name}.pt")
    torch.save(size_list, f"{TEST_RESULTS_DIR}/model_size/model_size_list_{mode_str}_{fusion_name}.pt")

    if REPLACE_BEST_LAYERS_SEQUANTIAL:
        best_layers = [x[0] for x in sorted(zip(LAYERS, accuracy_list[1:]), key=lambda x:-x[1])]
        print("Best layers:", best_layers)
        new_accuracy_list = [base_test_accuracy]
        new_size_list = [base_size]

        for layer in tqdm(best_layers):
            del model.model.layers[layer].mlp
            torch.cuda.empty_cache()
            model.model.layers[layer].mlp = experts_fusion(model, layer)
            new_accuracy_list.append(evaluate_qa(model, tokenizer, TEST_DATASET_NAME, TEST_SIZE, seed=SEED))
            new_size_list.append(count_params(model))
        print(f"model's accuracy list after {fusion_name} ({ReplaceMode.SEQ}):", new_accuracy_list)
        torch.save(new_accuracy_list, f"{TEST_RESULTS_DIR}/accuracy/accuracy_list_{ReplaceMode.SEQ}_{fusion_name}.pt")
        torch.save(new_size_list, f"{TEST_RESULTS_DIR}/model_size/model_size_list_{ReplaceMode.SEQ}_{fusion_name}.pt")
    free_model(model)
    model = None


if REPLACE_LAYERS_SMART_SEQUANTIAL:
    for fusion_name, experts_fusion in tqdm(list(experts_fusions.items())):
        changed = set()
        remain = set(LAYERS)
        print(f"Testing {fusion_name}")
        model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map=device, dtype=MODEL_DTYPE)
        if base_train_accuracy is None:
            base_train_accuracy = evaluate_qa(model, tokenizer, TRAIN_DATASET_NAME, TRAIN_SIZE, seed=SEED)
            base_test_accuracy = evaluate_qa(model, tokenizer, TEST_DATASET_NAME, TEST_SIZE, seed=SEED)
            base_size = count_params(model)
        accuracy_list = [base_test_accuracy]
        size_list = [base_size]
        for _ in tqdm(LAYERS):
            best_ind = -1
            best_score = -1

            for layer in remain:
                old_mlp = model.model.layers[layer].mlp
                model.model.layers[layer].mlp = experts_fusion(model, layer)
                new_score = evaluate_qa(model, tokenizer, TRAIN_DATASET_NAME, TRAIN_SIZE, seed=SEED)
                if new_score > best_score:
                    best_score = new_score
                    best_ind = layer
                del model.model.layers[layer].mlp
                torch.cuda.empty_cache()
                model.model.layers[layer].mlp = old_mlp
            remain.remove(best_ind)
            changed.add(best_ind)
            model.model.layers[best_ind].mlp = experts_fusion(model, best_ind)
            accuracy_list.append(evaluate_qa(model, tokenizer, TEST_DATASET_NAME, TEST_SIZE, seed=SEED))
            size_list.append(count_params(model))
        print(f"model's accuracy list after {fusion_name} {ReplaceMode.SMART_SEQ}:", accuracy_list)
        os.makedirs(f"{TEST_RESULTS_DIR}/accuracy", exist_ok=True)
        os.makedirs(f"{TEST_RESULTS_DIR}/model_size", exist_ok=True)
        torch.save(accuracy_list, f"{TEST_RESULTS_DIR}/accuracy/accuracy_list_{ReplaceMode.SMART_SEQ}_{fusion_name}.pt")
        torch.save(size_list, f"{TEST_RESULTS_DIR}/model_size/model_size_list_{ReplaceMode.SMART_SEQ}_{fusion_name}.pt")
        free_model(model)
        model = None

