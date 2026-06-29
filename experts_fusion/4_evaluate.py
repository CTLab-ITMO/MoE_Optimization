"""Stage 4 -- evaluation of the compressed model (§2.2, §3.4).

Collapses the old stages 7 (reconstruction benchmark), 8 (qualitative
generation) and 9 (final accuracy) into one stage that loads the base model
**once** (§3.4 load-once) and reuses it across every method by swapping the MoE
``mlp`` modules in and out, instead of the old stage 9's ``from_pretrained`` per
method (~15x model loads).

Load-once correctness: the original ``mlp`` modules are snapshotted once and the
model is fully restored between methods, so a method never sees another method's
replacement -- the parity property the load-once / load-per-method test checks.
The eval RNG is reset inside :func:`~common.qa_eval.evaluate_qa` per call.

Accuracy (the measured result, the golden gate) is computed first and written
under ``DATA_DIR`` (§1.4: no CWD-relative ``test_result_*``). The reconstruction
benchmark and the qualitative generation are best-effort diagnostics run
afterwards, so a fake/CPU-only quirk in a diagnostic never blocks the measured
results.
"""
import gc
import os
import time

import torch
import transformers
from tqdm import tqdm

from common import layout, provenance
from common.fusion_layer import get_fusion_layer_from_groups
from common.fusion import FUSION_REGISTRY
from common.metrics import METRIC_REGISTRY
from common.mlp_utils import FUSION_METHODS, best_metrics
from common.model_topology import discover_moe_layers
from common.names import BEST_PREFIX, TEST_RESULT_PREFIX, ZERO_FUSION, ReplaceMode
from common.plugins import load_plugins
from common.qa_eval import evaluate_qa
from common.registry import require_all_registered
from common.seeding import set_seed
from experts_fusion.config import (
    DATA_DIR,
    EXPERTS,
    FILTER_FUSIONS,
    FILTER_METRICS,
    INSPECT_FUSION_METHOD,
    LAYERS,
    LOAD_PLUGINS,
    MODEL_DTYPE,
    MODEL_NAME,
    QA_PARSER,
    REPLACE_BEST_LAYERS_SEQUANTIAL,
    REPLACE_LAYERS_SEQUANTIAL,
    REPLACE_LAYERS_SMART_SEQUANTIAL,
    ROUTER_ADAPTER,
    SEED,
    TEST_DATASET_NAME,
    TEST_SIZE,
    TOP_K,
    TRAIN_DATASET_NAME,
    TRAIN_SIZE,
    USE_DEFAULT_EXPERTS,
    config_snapshot_dict,
    device,
)

# Number of MLP inputs stacked per forward call in the recon benchmark, and how
# many such batches to time over.
BENCHMARK_BATCH_SIZE = 128
BENCHMARK_NUM_BATCHES = 100
# Qualitative generation knobs.
GENERATION_EXAMPLES = 10
MAX_NEW_TOKENS = 10

load_plugins(LOAD_PLUGINS)
require_all_registered(FILTER_FUSIONS, FUSION_REGISTRY, field="FILTER_FUSIONS")
require_all_registered(FILTER_METRICS, METRIC_REGISTRY, field="FILTER_METRICS")
provenance.assert_config_matches_snapshot(DATA_DIR, config_snapshot_dict())
layout.validate_layout_version(DATA_DIR)

fusion_methods = dict(FUSION_METHODS)
if FILTER_FUSIONS is not None:
    fusion_methods = {k: v for k, v in fusion_methods.items() if k in FILTER_FUSIONS}
selected_best_metrics = dict(best_metrics)
if FILTER_METRICS is not None:
    selected_best_metrics = {k: v for k, v in selected_best_metrics.items() if k in FILTER_METRICS}

set_seed(SEED)


def _dataset_tag(dataset):
    return f"{dataset['path'].split('/')[-1]}_{dataset['split']}"


RESULTS_DIR = os.path.join(
    DATA_DIR,
    f"{TEST_RESULT_PREFIX}{_dataset_tag(TRAIN_DATASET_NAME)}_{_dataset_tag(TEST_DATASET_NAME)}",
)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


class ZeroLayer(torch.nn.Module):
    """MLP layer that outputs zeros (a degenerate compression baseline)."""

    def forward(self, *args, **kwargs):
        return torch.zeros_like(args[0])


def _build_layer(name, layer, model):
    """Load the fusion layer for ``(name, layer)`` from the new versioned artifacts."""
    if name == ZERO_FUSION:
        return ZeroLayer().to(model.device, dtype=model.dtype)
    new_layer = get_fusion_layer_from_groups(
        layer,
        name,
        DATA_DIR,
        device=model.device,
        use_default_experts=USE_DEFAULT_EXPERTS,
        num_experts=EXPERTS,
        router_adapter=ROUTER_ADAPTER,
        top_k=TOP_K,
    ).to(model.device, dtype=model.dtype)
    for expert in new_layer.fusion_experts:
        if expert.U is not None:
            expert.U = expert.U.to(model.device, dtype=model.dtype)
    return new_layer


# Names evaluated: every (filtered) fusion method, the zero baseline, and the
# per-metric Best_* pseudo-methods written by stage 3.
eval_names = list(fusion_methods.keys()) + [ZERO_FUSION]
eval_names += [f"{BEST_PREFIX}{metric}" for metric in selected_best_metrics]


def _eval_train(model, tokenizer):
    return evaluate_qa(model, tokenizer, TRAIN_DATASET_NAME, TRAIN_SIZE, seed=SEED, parser=QA_PARSER)


def _eval_test(model, tokenizer):
    return evaluate_qa(model, tokenizer, TEST_DATASET_NAME, TEST_SIZE, seed=SEED, parser=QA_PARSER)


def _save_list(values, mode, name, kind):
    out_dir = os.path.join(RESULTS_DIR, kind)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(values, os.path.join(out_dir, f"{kind}_list_{mode}_{name}.pt"))


def accuracy(model, tokenizer, moe_layers, originals):
    """Reproduce the old stage 9 accuracy sweeps on the single, load-once model."""

    def restore_all():
        for layer in moe_layers:
            model.model.layers[layer].mlp = originals[layer]

    base_train = _eval_train(model, tokenizer)
    base_test = _eval_test(model, tokenizer)
    base_size = count_params(model)

    for name in eval_names:
        restore_all()
        print(f"Testing {name}")
        accuracy_list = [base_train]
        size_list = [base_size]
        for layer in tqdm(moe_layers):
            model.model.layers[layer].mlp = _build_layer(name, layer, model)
            accuracy_list.append(_eval_train(model, tokenizer))
            size_list.append(count_params(model))
            if not REPLACE_LAYERS_SEQUANTIAL:
                model.model.layers[layer].mlp = originals[layer]

        mode = ReplaceMode.SEQ if REPLACE_LAYERS_SEQUANTIAL else ReplaceMode.SINGLE
        print(f"accuracy list after {name} ({mode}):", accuracy_list)
        _save_list(accuracy_list, mode, name, "accuracy")
        _save_list(size_list, mode, name, "model_size")

        if REPLACE_BEST_LAYERS_SEQUANTIAL:
            restore_all()
            best_layers = [
                lyr for lyr, _ in sorted(
                    zip(moe_layers, accuracy_list[1:]), key=lambda pair: -pair[1]
                )
            ]
            best_accuracy = [base_test]
            best_size = [base_size]
            for layer in tqdm(best_layers):
                model.model.layers[layer].mlp = _build_layer(name, layer, model)
                best_accuracy.append(_eval_test(model, tokenizer))
                best_size.append(count_params(model))
            print(f"accuracy list after {name} ({ReplaceMode.SEQ}):", best_accuracy)
            _save_list(best_accuracy, ReplaceMode.SEQ, name, "accuracy")
            _save_list(best_size, ReplaceMode.SEQ, name, "model_size")

    restore_all()

    if REPLACE_LAYERS_SMART_SEQUANTIAL:
        _smart_accuracy(model, tokenizer, moe_layers, originals, base_test, base_size)


def _smart_accuracy(model, tokenizer, moe_layers, originals, base_test, base_size):
    """Greedy best-next-layer accuracy sweep (old stage 9 smart-seq block)."""

    def restore_all():
        for layer in moe_layers:
            model.model.layers[layer].mlp = originals[layer]

    for name in tqdm(eval_names):
        restore_all()
        print(f"Testing {name}")
        remaining = set(moe_layers)
        accuracy_list = [base_test]
        size_list = [base_size]
        for _ in tqdm(moe_layers):
            best_layer, best_score = -1, -1
            for layer in remaining:
                model.model.layers[layer].mlp = _build_layer(name, layer, model)
                score = _eval_train(model, tokenizer)
                if score > best_score:
                    best_score, best_layer = score, layer
                model.model.layers[layer].mlp = originals[layer]
            remaining.remove(best_layer)
            model.model.layers[best_layer].mlp = _build_layer(name, best_layer, model)
            accuracy_list.append(_eval_test(model, tokenizer))
            size_list.append(count_params(model))
        print(f"accuracy list after {name} {ReplaceMode.SMART_SEQ}:", accuracy_list)
        _save_list(accuracy_list, ReplaceMode.SMART_SEQ, name, "accuracy")
        _save_list(size_list, ReplaceMode.SMART_SEQ, name, "model_size")
    restore_all()


def diagnostics(model, tokenizer, moe_layers, originals):
    """Best-effort qualitative diagnostics (recon benchmark + generation).

    Run after the measured accuracy is already written; a fake/CPU-only quirk
    here must not block the gate, so failures are reported, not raised.
    """

    def restore_all():
        for layer in moe_layers:
            model.model.layers[layer].mlp = originals[layer]

    try:
        _reconstruction_benchmark(model, moe_layers)
    except Exception as exc:  # diagnostics only -- never blocks the measured result
        print(f"[diagnostics] reconstruction benchmark skipped: {exc!r}")
    restore_all()
    try:
        _generation(model, tokenizer, moe_layers)
    except Exception as exc:  # diagnostics only -- never blocks the measured result
        print(f"[diagnostics] generation skipped: {exc!r}")
    restore_all()


def _reconstruction_benchmark(model, moe_layers):
    layer = moe_layers[0]
    train_data = torch.load(layout.mlp_inputs_path(DATA_DIR, layout.TRAIN_SPLIT, layer))
    fusion = get_fusion_layer_from_groups(
        layer,
        INSPECT_FUSION_METHOD,
        DATA_DIR,
        device=device,
        use_default_experts=True,
        num_experts=EXPERTS,
        router_adapter=ROUTER_ADAPTER,
        top_k=TOP_K,
    )
    total = min(BENCHMARK_NUM_BATCHES, len(train_data) // BENCHMARK_BATCH_SIZE)
    errors = []
    with torch.no_grad():
        for start in range(0, BENCHMARK_BATCH_SIZE * total, BENCHMARK_BATCH_SIZE):
            batch = torch.stack(train_data[start : start + BENCHMARK_BATCH_SIZE]).unsqueeze(0).to(device)
            fused_out = fusion.forward(batch)[0]
            base_out = fusion.forward_default(batch)[0]
            errors.append((fused_out - base_out).square().sum().sqrt().item())
    if errors:
        print(f"[diagnostics] recon error (layer {layer}, {INSPECT_FUSION_METHOD}): {errors[0]:.4f}")


def _generation(model, tokenizer, moe_layers):
    import datasets

    from experts_fusion.config import DATASET_TEXT_FIELD, VALIDATION_DATASET_NAME

    validation = datasets.load_dataset(**VALIDATION_DATASET_NAME)[DATASET_TEXT_FIELD][:GENERATION_EXAMPLES]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        for text in validation
    ]
    base_generations = [
        tokenizer.decode(model.generate(**prompt, max_new_tokens=MAX_NEW_TOKENS)[0])
        for prompt in prompts
    ]
    for layer in moe_layers:
        model.model.layers[layer].mlp = _build_layer(INSPECT_FUSION_METHOD, layer, model)
    fused_generations = [
        tokenizer.decode(model.generate(**prompt, max_new_tokens=MAX_NEW_TOKENS)[0])
        for prompt in prompts
    ]
    for before, after in zip(base_generations, fused_generations):
        print(before, "\n" + "+" * 40 + "\n", after, "\n" + "-" * 40)


def main():
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map=device, dtype=MODEL_DTYPE
    )
    moe_layers = discover_moe_layers(model, selected=LAYERS)
    # Snapshot the original mlp modules once; load-once relies on restoring them
    # between methods rather than reloading the model.
    originals = {layer: model.model.layers[layer].mlp for layer in moe_layers}

    accuracy(model, tokenizer, moe_layers, originals)
    diagnostics(model, tokenizer, moe_layers, originals)

    try:
        model.cpu()
    except Exception:
        pass
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


main()
