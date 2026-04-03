#!/usr/bin/env python
# coding: utf-8

import os
import sys
import torch
import transformers
from datasets import load_dataset
from matplotlib import pyplot as plt
from tqdm import tqdm
from config import MODEL_NAME, LAYERS, VALIDATION_SIZE, device, REPLACE_LAYERS_SEQUANTIAL, DATA_DIR, REPLACE_BEST_LAYERS_SEQUANTIAL

sys.path.append("..")
from common.fusion_layer import get_fusion_layer_by_index
from common.mlp_utils import FUSION_METHODS

model_id = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
use_default_experts = False


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

@torch.no_grad()
def test_model(model):
    correct_count = 0
    total_count = 0

    for sample in tqdm(eval_dataset, desc="testing model"):
        prompt = format_prompt(sample)
        ground_truth = sample['answerKey']

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=False
            )

        generated_token_id = outputs[0][inputs.input_ids.shape[1]:]
        response = tokenizer.decode(
            generated_token_id,
            skip_special_tokens=True
        ).strip().upper()

        valid_answers = ['A', 'B', 'C', 'D', 'E']
        clean_response = response[0] if (len(response) > 0
                                         and response[0] in valid_answers
                                         ) else response

        is_correct = (clean_response == ground_truth)
        if is_correct:
            correct_count += 1

        total_count += 1

        # print(f"Q: {sample['question'][:30]}... | Pred: {clean_response} | True: {ground_truth}")

    accuracy = correct_count / total_count
    print("\n--- Results ---")
    print(f"Total examples: {total_count}")
    print(f"Correct answers: {correct_count}")
    print(f"Accuracy: {accuracy:.2%}")
    return accuracy


base_accuracy = None


class ZeroLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        return torch.zeros_like(args[0])

def zero_fusion(model, layer):
    return ZeroLayer().to(model.device, dtype=model.dtype)


def default_fusion(name):
    def inner(model, layer):
        return get_fusion_layer_by_index(layer, f"fusion_experts/{name}", device=model.device,
                                            use_default_experts=use_default_experts
                                            ).to(model.device, dtype=model.dtype)
    return inner


experts_fusions = {name:default_fusion(name) for name in FUSION_METHODS.keys()}
experts_fusions["zero"] = zero_fusion


for fusion_name, experts_fusion in experts_fusions.items():
    print(f"Testing {fusion_name}")
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map=device)
    model.config.use_cache = False
    model.generation_config.use_cache = False

    if base_accuracy is None:
        base_accuracy = test_model(model)
    accuracy_list = [base_accuracy]

    for layer in tqdm(LAYERS):
        if REPLACE_LAYERS_SEQUANTIAL:
            del model.model.layers[layer].mlp
            torch.cuda.empty_cache()
            model.model.layers[layer].mlp = experts_fusion(model, layer)
            accuracy_list.append(test_model(model))
        else:
            old_mlp = model.model.layers[layer].mlp
            model.model.layers[layer].mlp = experts_fusion(model, layer)
            accuracy_list.append(test_model(model))
            del model.model.layers[layer].mlp
            torch.cuda.empty_cache()
            model.model.layers[layer].mlp = old_mlp

    print(f"model's accuracy list after {fusion_name} ({'seq' if REPLACE_LAYERS_SEQUANTIAL else 'single'}):", accuracy_list)
    os.makedirs("accuracy", exist_ok=True)
    torch.save(accuracy_list, f"accuracy/accuracy_list_{'seq' if REPLACE_LAYERS_SEQUANTIAL else 'single'}_{fusion_name}.pt")

    if REPLACE_BEST_LAYERS_SEQUANTIAL:
        best_layers = [x[0] for x in sorted(enumerate(accuracy_list[1:]), key=lambda x:-x[1])]
        print("Best layers:", best_layers)
        new_accuracy_list = [base_accuracy]

        for layer in tqdm(best_layers):
            del model.model.layers[layer].mlp
            torch.cuda.empty_cache()
            model.model.layers[layer].mlp = experts_fusion(model, layer)
            new_accuracy_list.append(test_model(model))
        print(f"model's accuracy list after {fusion_name} (seq):", new_accuracy_list)
        torch.save(new_accuracy_list, f"accuracy/accuracy_list_seq_{fusion_name}.pt")
    del model

