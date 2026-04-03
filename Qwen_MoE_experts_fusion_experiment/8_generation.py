#!/usr/bin/env python
# coding: utf-8

import torch
from tqdm import tqdm
import sys
import transformers
import datasets
from config import LAYERS, MODEL_NAME, TEST_DATASET_NAME, device, DATA_DIR

sys.path.append("..")
from common.fusion_layer import get_fusion_layer_by_index

EXAMPLES = 10


if "model" not in globals():
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_NAME)
if "tokenizer" not in globals():
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
model = model.to(device)


if "_test_data" not in globals():
    _test_data = datasets.load_dataset(**TEST_DATASET_NAME)["question"]
test_data = _test_data[:EXAMPLES]


@torch.autocast("cuda")
def generate_examples(model, examples):
    res = []
    for example in tqdm(examples):
        outputs = model.generate(**example, max_new_tokens=10)
        res.append((tokenizer.decode(outputs[0])))
    return res


examples = []
for ind in range(EXAMPLES):
    messages = [
        {"role": "user", "content": test_data[ind]},
    ]
    examples.append(tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device))


base_examples = generate_examples(model, examples)


for layer in LAYERS:
    del model.model.layers[layer].mlp
    torch.cuda.empty_cache()
    new_mlp = get_fusion_layer_by_index(
        layer,
        f"{DATA_DIR}/fusion_experts/ZipIt",
        device=device,
        use_default_experts=False
    ).to(device)
    model.model.layers[layer].mlp = new_mlp


new_examples = generate_examples(model, examples)


for base_generation, new_generation in zip(base_examples, new_examples):
    print(
        base_generation,
        "\n" + "+" * 100 + "\n",
        new_generation,
        "\n" + "-" * 100 + "\n"
    )

