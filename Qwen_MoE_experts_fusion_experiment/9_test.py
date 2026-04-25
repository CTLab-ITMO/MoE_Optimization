#!/usr/bin/env python
# coding: utf-8

import os
import sys
import torch
import transformers
from datasets import load_dataset
from matplotlib import pyplot as plt
from tqdm import tqdm
from config import MODEL_NAME, LAYERS, VALIDATION_SIZE, device, REPLACE_LAYERS_SEQUANTIAL, DATA_DIR, REPLACE_BEST_LAYERS_SEQUANTIAL, FILTER_FUSIONS, FILTER_METRICS, REPLACE_LAYERS_SMART_SEQUANTIAL

sys.path.append("..")
from common.fusion_layer import get_fusion_layer_by_index
from common.mlp_utils import FUSION_METHODS, best_metrics

model_id = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
use_default_experts = False

if FILTER_FUSIONS is not None:
    FUSION_METHODS = {k:v for k, v in FUSION_METHODS.items() if k in FILTER_FUSIONS}
if FILTER_METRICS is not None:
    best_metrics = {k:v for k, v in best_metrics.items() if k in FILTER_METRICS}


tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)


dataset = load_dataset("commonsense_qa", split="validation")

print(f"Samples in dataset: {len(dataset)}")

print("Data example:", dataset[0])


def format_prompt(sample):
    question = sample['question']
    choices = sample['choices']

    options_text = ""
    for label, text in zip(choices['label'], choices['text']):
        options_text += f"{label}. {text}\n"

    system_message = (
        "You are a precise evaluation system. "
        "You will be given a question and multiple-choice options. "
        "Your task is to answer with ONLY the letter corresponding to the correct option (A, B, C, D, or E). "
        "Do not write the full answer, do not write explanations. Output only one character."
    )

    user_message = f"Question: {question}\nOptions:\n{options_text}\nAnswer:"

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    text_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    return text_prompt


print("Prompt example:")
print(format_prompt(dataset[0]))


validation_size = min(len(dataset), VALIDATION_SIZE)
eval_dataset = dataset.select(range(validation_size))

prompts = []
ground_truths = []

for sample in eval_dataset:
    prompt = format_prompt(sample)
    ground_truth = sample['answerKey']
    prompts.append(prompt)
    ground_truths.append(ground_truth)
tokenizer.padding_side = 'left'


answer_letters = ['A', 'B', 'C', 'D', 'E']
answer_token_ids = []

for letter in answer_letters:
    token_ids = tokenizer.encode(f" {letter}", add_special_tokens=False)
    if token_ids:
        token_id = token_ids[0]
    else:
        token_ids = tokenizer.encode(letter, add_special_tokens=False)
        if token_ids:
            token_id = token_ids[0]
        else:
            print(f"Warning: Could not find token for {letter}")
            token_id = -1
    answer_token_ids.append(token_id)
letter_to_idx = {letter: idx for idx, letter in enumerate(answer_letters)}

@torch.no_grad()
def test_model(model, batch_size=100):
    correct_count = 0
    total_count = 0


    answer_token_tensor = torch.tensor(answer_token_ids, device=model.device)  # [5]


    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        batch_truths = ground_truths[i:i+batch_size]

        inputs = tokenizer(
            batch_prompts, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=512
        ).to(model.device)

        outputs = model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            output_hidden_states=False,
            output_attentions=False,
            use_cache=False
        )

        last_token_logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]

        answer_indices = answer_token_tensor.unsqueeze(0).expand(last_token_logits.size(0), -1)
        answer_logits = torch.gather(last_token_logits, 1, answer_indices)  # [batch_size, 5]

        predicted_indices = torch.argmax(answer_logits, dim=1)  # [batch_size]

        truth_indices = torch.tensor(
            [letter_to_idx[truth] for truth in batch_truths],
            device=model.device
        )  # [batch_size]

        correct_mask = predicted_indices == truth_indices  # [batch_size]
        correct_count += correct_mask.sum().item()
        total_count += len(batch_prompts)

        del inputs, outputs, last_token_logits, answer_indices, answer_logits, predicted_indices, truth_indices, correct_mask
        torch.cuda.empty_cache()

    accuracy = correct_count / total_count
    return accuracy


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
                                            use_default_experts=use_default_experts
                                            ).to(model.device, dtype=model.dtype)
        for expert in new_layer.fusion_experts:
            if expert.U is not None:
                expert.U = expert.U.to(model.device, dtype=model.dtype)
        return new_layer

    return inner


experts_fusions = {name:default_fusion(name) for name in FUSION_METHODS.keys()}
experts_fusions["zero"] = zero_fusion
for best_metric in best_metrics.keys():
    experts_fusions[f"Best_{best_metric}"] = default_fusion(f"Best_{best_metric}")


for fusion_name, experts_fusion in experts_fusions.items():
    print(f"Testing {fusion_name}")
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map=device)

    if base_accuracy is None:
        base_accuracy = test_model(model)
        base_size = count_params(model)
    accuracy_list = [base_accuracy]
    size_list = [base_size]

    for layer in tqdm(LAYERS):
        if REPLACE_LAYERS_SEQUANTIAL:
            del model.model.layers[layer].mlp
            torch.cuda.empty_cache()
            model.model.layers[layer].mlp = experts_fusion(model, layer)
            accuracy_list.append(test_model(model))
            size_list.append(count_params(model))
        else:
            old_mlp = model.model.layers[layer].mlp
            model.model.layers[layer].mlp = experts_fusion(model, layer)
            accuracy_list.append(test_model(model))
            size_list.append(count_params(model))
            del model.model.layers[layer].mlp
            torch.cuda.empty_cache()
            model.model.layers[layer].mlp = old_mlp

    mode_str = 'seq' if REPLACE_LAYERS_SEQUANTIAL else 'single'
    print(f"model's accuracy list after {fusion_name} ({mode_str}):", accuracy_list)
    os.makedirs("test_results/accuracy", exist_ok=True)
    os.makedirs("test_results/model_size", exist_ok=True)
    torch.save(accuracy_list, f"test_results/accuracy/accuracy_list_{mode_str}_{fusion_name}.pt")
    torch.save(size_list, f"test_results/model_size/model_size_list_{mode_str}_{fusion_name}.pt")

    if REPLACE_BEST_LAYERS_SEQUANTIAL:
        best_layers = [x[0] for x in sorted(enumerate(accuracy_list[1:]), key=lambda x:-x[1])]
        print("Best layers:", best_layers)
        new_accuracy_list = [base_accuracy]
        new_size_list = [base_size]

        for layer in tqdm(best_layers):
            del model.model.layers[layer].mlp
            torch.cuda.empty_cache()
            model.model.layers[layer].mlp = experts_fusion(model, layer)
            new_accuracy_list.append(test_model(model))
            new_size_list.append(count_params(model))
        print(f"model's accuracy list after {fusion_name} (seq):", new_accuracy_list)
        torch.save(new_accuracy_list, f"test_results/accuracy/accuracy_list_seq_{fusion_name}.pt")
        torch.save(new_size_list, f"test_results/model_size/model_size_list_seq_{fusion_name}.pt")
    del model


if REPLACE_LAYERS_SMART_SEQUANTIAL:
    for fusion_name, experts_fusion in tqdm(list(experts_fusions.items())):
        changed = set()
        remain = set(LAYERS)
        print(f"Testing {fusion_name}")
        model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map=device)
        if base_accuracy is None:
            base_accuracy = test_model(model)
            base_size = count_params(model)
        accuracy_list = [base_accuracy]
        size_list = [base_size]
        for _ in tqdm(LAYERS):
            best_ind = -1
            best_score = -1

            for layer in remain:
                old_mlp = model.model.layers[layer].mlp
                model.model.layers[layer].mlp = experts_fusion(model, layer)
                new_score = test_model(model)
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
        os.makedirs("test_results/accuracy", exist_ok=True)
        os.makedirs("test_results/model_size", exist_ok=True)
        torch.save(accuracy_list, f"test_results/accuracy/accuracy_list_smart_seq_{fusion_name}.pt")
        torch.save(size_list, f"test_results/model_size/model_size_list_smart_seq_{fusion_name}.pt")
        del model

