#!/usr/bin/env python
# coding: utf-8

import sys
import torch
import transformers
from datasets import load_dataset
from matplotlib import pyplot as plt
from tqdm import tqdm
from config import MODEL_NAME, LAYERS, VALIDATION_SIZE, device, EVAL_MODEL_AFTER_EVERY_STEP

sys.path.append("..")
from common.fusion_layer import get_fusion_layer_by_index

model_id = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
use_default_experts = False


if "model" not in globals():
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map=device)
if "tokenizer" not in globals():
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)

model.config.use_cache = False
model.generation_config.use_cache = False


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


accuracy_list = [test_model(model)]
base_accuracy = accuracy_list[0]


for layer in tqdm(LAYERS):
    del model.model.layers[layer].mlp
    torch.cuda.empty_cache()
    new_mlp = get_fusion_layer_by_index(layer, device=model.device,
                                        use_default_experts=use_default_experts
                                        ).to(model.device)
    model.model.layers[layer].mlp = new_mlp
    if EVAL_MODEL_AFTER_EVERY_STEP:
        accuracy_list.append(test_model(model))

if not EVAL_MODEL_AFTER_EVERY_STEP:
    accuracy_list.append(test_model(model))


new_accuracy = accuracy_list[-1]


print("Base model's accuracy:", base_accuracy)
print("New model's accuracy:", new_accuracy)


print("Base model's correct answers:", base_accuracy * validation_size)
print("New model's correct answers:", new_accuracy * validation_size)


print("model's accuracy list:", accuracy_list)
torch.save(accuracy_list, "accuracy_list.pt")

