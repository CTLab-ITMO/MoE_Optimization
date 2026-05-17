import os
import random
import re
import gc
import warnings
import subprocess
import sys
from collections import defaultdict
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging
from tqdm import tqdm
import argparse

warnings.filterwarnings("ignore")
logging.set_verbosity_error()

# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6"

SEED = 1235
BIAS_START_LAYER = 50
POSITIVE_BIAS_LAYERS = None  #[2,4,14,17] #[4,6,8,13]chem #[1,2,7,9,13,14]biology #[0,9,10,13,14,23]  #[2,4,14,17,21]  #[9,13,20,22,23]-physics #[0,9,10,14,23]-history #[13,20,22] physics #[2,4,14] math None = auto by threshold, список = manual.
NEGATIVE_BIAS_LAYERS = None  #[5]  #[2,17]chem #[3,12,15,18] #[0,12,21]  #[3,4,5,12,16]  #[2,3,17,18,19]-physics  #[0,8,12,19,21] history #[2,17,19] physics #[4,5,16] math None = auto by threshold, список = manual.
NUM_EXP_POS_BIAS_APPLY = 3
NUM_EXP_NEG_BIAS_APPLY = 3
POSITIVE_TOP3_MEAN_THRESHOLD = 0.6
NEGATIVE_TOP3_ABS_MEAN_THRESHOLD = 0.6
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
#

# print("[DEBUG] Loading MMLU datasets...")
data_folder = "/home/anmilka/MoE/mmlu_data"

os.makedirs(data_folder, exist_ok=True)

saved_paths = {
    "all_part": os.path.join(data_folder, "test_samples_all_part"),
    "all": os.path.join(data_folder, "test_samples_all"),
    "alg": os.path.join(data_folder, "test_samples_alg"),
    "math": os.path.join(data_folder, "test_samples_math"),
    "elem_math": os.path.join(data_folder, "test_samples_elem_math"),
    "high_school_math": os.path.join(data_folder, "test_samples_high_school_math"),
    "high_school_physics": os.path.join(data_folder, "test_samples_high_school_physics"),
    "conceptual_physics": os.path.join(data_folder, "test_samples_conceptual_physics"),
    "college_physics": os.path.join(data_folder, "test_samples_college_physics"),
    "high_school_biology": os.path.join(data_folder, "test_samples_high_school_biology"),
    "college_biology": os.path.join(data_folder, "test_samples_college_biology"),
    "college_biology_part": os.path.join(data_folder, "test_samples_college_biology_part"),
    "high_school_chemistry": os.path.join(data_folder, "test_samples_high_school_chemistry"),
    "college_chemistry": os.path.join(data_folder, "test_samples_college_chemistry"),
    "college_chemistry_part": os.path.join(data_folder, "test_samples_college_chemistry_part"),
    "high_school_us_history": os.path.join(data_folder, "test_samples_high_school_us_history"),
    "high_school_european_history": os.path.join(data_folder, "test_samples_high_school_european_history"),
    "high_school_world_history": os.path.join(data_folder, "test_samples_high_school_world_history"),
    "high_school_world_history_part": os.path.join(data_folder, "test_samples_high_school_world_history_part"),
    "prehistory": os.path.join(data_folder, "test_samples_prehistory"),
    "college_med": os.path.join(data_folder, "test_samples_college_med"),
    "prof_med": os.path.join(data_folder, "test_samples_prof_med"),
    "clin_know": os.path.join(data_folder, "test_samples_clin_know"),
}

all_exist = all(os.path.exists(p) for p in saved_paths.values())
#all_exist = None

if all_exist:
#     print("[DEBUG] Loading datasets from saved files...")
    from datasets import load_from_disk
    test_samples_all_part = load_from_disk(saved_paths["all_part"])
    test_samples_all = load_from_disk(saved_paths["all"])
    test_samples_alg = load_from_disk(saved_paths["alg"])
    test_samples_math = load_from_disk(saved_paths["math"])
    test_samples_elem_math = load_from_disk(saved_paths["elem_math"])
    test_samples_high_school_math = load_from_disk(saved_paths["high_school_math"])
    test_samples_high_school_physics = load_from_disk(saved_paths["high_school_physics"])
    test_samples_conceptual_physics = load_from_disk(saved_paths["conceptual_physics"])
    test_samples_college_physics = load_from_disk(saved_paths["college_physics"])
    test_samples_high_school_biology = load_from_disk(saved_paths["high_school_biology"])
    test_samples_college_biology = load_from_disk(saved_paths["college_biology"])
    test_samples_college_biology_part = load_from_disk(saved_paths["college_biology_part"])
    test_samples_high_school_chemistry = load_from_disk(saved_paths["high_school_chemistry"])
    test_samples_college_chemistry = load_from_disk(saved_paths["college_chemistry"])
    test_samples_college_chemistry_part = load_from_disk(saved_paths["college_chemistry_part"])
    test_samples_high_school_us_history = load_from_disk(saved_paths["high_school_us_history"])
    test_samples_high_school_european_history = load_from_disk(saved_paths["high_school_european_history"])
    test_samples_high_school_world_history = load_from_disk(saved_paths["high_school_world_history"])
    test_samples_high_school_world_history_part = load_from_disk(saved_paths["high_school_world_history_part"])
    test_samples_prehistory = load_from_disk(saved_paths["prehistory"])
    test_samples_college_med = load_from_disk(saved_paths["college_med"])
    test_samples_prof_med = load_from_disk(saved_paths["prof_med"])
    test_samples_clin_know = load_from_disk(saved_paths["clin_know"])
#     print("[DEBUG] All datasets loaded from saved files")
else:
#     print("[DEBUG] Downloading and preparing datasets...")
    mmlu_dataset = load_dataset("cais/mmlu", "all")
    test_samples_all_part = mmlu_dataset["test"].shuffle(seed=SEED).select(range(100))
    test_samples_all = mmlu_dataset["test"].shuffle(seed=SEED)

    mmlu_dataset = load_dataset("cais/mmlu", "abstract_algebra")
    test_samples_alg = mmlu_dataset["test"].shuffle(seed=SEED)

    mmlu_dataset = load_dataset("cais/mmlu", "college_mathematics")
    test_samples_math = mmlu_dataset["test"].shuffle(seed=SEED)

    mmlu_dataset = load_dataset("cais/mmlu", "elementary_mathematics")
    test_samples_elem_math = mmlu_dataset["test"].shuffle(seed=SEED)

    mmlu_dataset = load_dataset("cais/mmlu", "high_school_mathematics")
    test_samples_high_school_math = mmlu_dataset["test"].shuffle(seed=SEED)

    mmlu_dataset = load_dataset("cais/mmlu", "high_school_physics")
    test_samples_high_school_physics = mmlu_dataset["test"].shuffle(seed=SEED)

    mmlu_dataset = load_dataset("cais/mmlu", "conceptual_physics")
    test_samples_conceptual_physics = mmlu_dataset["test"].shuffle(seed=SEED)

    mmlu_dataset = load_dataset("cais/mmlu", "college_physics")
    test_samples_college_physics = mmlu_dataset["test"].shuffle(seed=SEED).select(range(100))

    mmlu_dataset = load_dataset("cais/mmlu", "high_school_biology")
    test_samples_high_school_biology = mmlu_dataset["test"].shuffle(seed=SEED)

    mmlu_dataset = load_dataset("cais/mmlu", "college_biology")
    test_samples_college_biology = mmlu_dataset["test"].shuffle(seed=SEED)
    test_samples_college_biology_part = mmlu_dataset["test"].shuffle(seed=SEED).select(range(100))

    mmlu_dataset = load_dataset("cais/mmlu", "high_school_chemistry")
    test_samples_high_school_chemistry = mmlu_dataset["test"].shuffle(seed=SEED)

    mmlu_dataset = load_dataset("cais/mmlu", "college_chemistry")
    test_samples_college_chemistry = mmlu_dataset["test"].shuffle(seed=SEED)
    test_samples_college_chemistry_part = mmlu_dataset["test"].shuffle(seed=SEED).select(range(100))

    mmlu_dataset = load_dataset("cais/mmlu", "high_school_us_history")
    test_samples_high_school_us_history = mmlu_dataset["test"].shuffle(seed=SEED)

    mmlu_dataset = load_dataset("cais/mmlu", "high_school_european_history")
    test_samples_high_school_european_history = mmlu_dataset["test"].shuffle(seed=SEED)

    mmlu_dataset = load_dataset("cais/mmlu", "high_school_world_history")
    test_samples_high_school_world_history = mmlu_dataset["test"].shuffle(seed=SEED)
    test_samples_high_school_world_history_part = mmlu_dataset["test"].shuffle(seed=SEED).select(range(100))

    mmlu_dataset = load_dataset("cais/mmlu", "prehistory")
    test_samples_prehistory = mmlu_dataset["test"].shuffle(seed=SEED)

    mmlu_dataset = load_dataset("cais/mmlu", "college_medicine")
    test_samples_college_med = mmlu_dataset["test"].shuffle(seed=SEED)

    mmlu_dataset = load_dataset("cais/mmlu", "professional_medicine")
    test_samples_prof_med = mmlu_dataset["test"].shuffle(seed=SEED)

    mmlu_dataset = load_dataset("cais/mmlu", "clinical_knowledge")
    test_samples_clin_know = mmlu_dataset["test"].shuffle(seed=SEED)
#
#     print("[DEBUG] Saving datasets to disk...")
    test_samples_all_part.save_to_disk(saved_paths["all_part"])
    test_samples_all.save_to_disk(saved_paths["all"])
    test_samples_alg.save_to_disk(saved_paths["alg"])
    test_samples_math.save_to_disk(saved_paths["math"])
    test_samples_elem_math.save_to_disk(saved_paths["elem_math"])
    test_samples_high_school_math.save_to_disk(saved_paths["high_school_math"])
    test_samples_high_school_physics.save_to_disk(saved_paths["high_school_physics"])
    test_samples_conceptual_physics.save_to_disk(saved_paths["conceptual_physics"])
    test_samples_college_physics.save_to_disk(saved_paths["college_physics"])
    test_samples_high_school_biology.save_to_disk(saved_paths["high_school_biology"])
    test_samples_college_biology.save_to_disk(saved_paths["college_biology"])
    test_samples_college_biology_part.save_to_disk(saved_paths["college_biology_part"])
    test_samples_high_school_chemistry.save_to_disk(saved_paths["high_school_chemistry"])
    test_samples_college_chemistry.save_to_disk(saved_paths["college_chemistry"])
    test_samples_college_chemistry_part.save_to_disk(saved_paths["college_chemistry_part"])
    test_samples_high_school_us_history.save_to_disk(saved_paths["high_school_us_history"])
    test_samples_high_school_european_history.save_to_disk(saved_paths["high_school_european_history"])
    test_samples_high_school_world_history.save_to_disk(saved_paths["high_school_world_history"])
    test_samples_high_school_world_history_part.save_to_disk(saved_paths["high_school_world_history_part"])
    test_samples_prehistory.save_to_disk(saved_paths["prehistory"])
    test_samples_college_med.save_to_disk(saved_paths["college_med"])
    test_samples_prof_med.save_to_disk(saved_paths["prof_med"])
    test_samples_clin_know.save_to_disk(saved_paths["clin_know"])
#     print("[DEBUG] All datasets saved to disk")


# abstract_algebra -> abstract algebra
def pretty_subject(s: str) -> str:
    return s.replace("_", " ")

def format_mmlu_question(example):
    subject = pretty_subject(example.get("subject"))
    question = example["question"]
    choices = example["choices"]
    hint = f'There is a single choice question about {subject}. Answer the question by replying only with letter A, B, C or D.'
    return (
        f"{hint}\n"
        f"Question: {question}\n"
        f"A. {choices[0]}\n"
        f"B. {choices[1]}\n"
        f"C. {choices[2]}\n"
        f"D. {choices[3]}\n"
        f"Answer:\n"
    )

def build_chat_texts(tokenizer, prompts):
    chat_texts = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        chat_texts.append(text)
    return chat_texts


def first_option_postprocess(text: str, options: str = 'ABCD', cushion=True) -> str:  # noqa: E501
    """Find first valid option for text."""

    # yapf: disable
    # flake8: noqa: W605
    patterns = [
        # f'з­”жЎ€жЇ?\s*([{options}])',
        # f'з­”жЎ€жЇ?\s*пјљ\s*([{options}])',
        # f'з­”жЎ€жЇ?\s*:\s*([{options}])',
        # f'з­”жЎ€йЂ‰йЎ№еє”?иЇҐ?жЇ\s*([{options}])',
        # f'з­”жЎ€йЂ‰йЎ№еє”?иЇҐ?дёє\s*([{options}])',
        # f'з­”жЎ€еє”иЇҐ?жЇ\s*([{options}])',
        # f'з­”жЎ€еє”иЇҐ?йЂ‰\s*([{options}])',
        # f'з­”жЎ€йЂ‰йЎ№дёє?\s*пјљ\s*([{options}])',
        # f'з­”жЎ€йЂ‰йЎ№дёє?\s+\(?\*?\*?([{options}])\*?\*?\)?',
        # f'з­”жЎ€йЂ‰йЎ№жЇ?\s*:\s*([{options}])',
        # f'з­”жЎ€дёє\s*([{options}])',
        # f'з­”жЎ€йЂ‰\s*([{options}])',
        # f'йЂ‰ж‹©?\s*([{options}])',
        # f'ж•…йЂ‰?\s*([{options}])'
        # f'еЏЄжњ‰йЂ‰?йЎ№?\s?([{options}])\s?жЇ?еЇ№',
        # f'еЏЄжњ‰йЂ‰?йЎ№?\s?([{options}])\s?жЇ?й”™',
        # f'еЏЄжњ‰йЂ‰?йЎ№?\s?([{options}])\s?дёЌ?ж­ЈзЎ®',
        # f'еЏЄжњ‰йЂ‰?йЎ№?\s?([{options}])\s?й”™иЇЇ',
        # f'иЇґжі•дёЌ?еЇ№йЂ‰?йЎ№?зљ„?жЇ\s?([{options}])',
        # f'иЇґжі•дёЌ?ж­ЈзЎ®йЂ‰?йЎ№?зљ„?жЇ\s?([{options}])',
        # f'иЇґжі•й”™иЇЇйЂ‰?йЎ№?зљ„?жЇ\s?([{options}])',
        # f'([{options}])\s?жЇж­ЈзЎ®зљ„',
        # f'([{options}])\s?жЇж­ЈзЎ®з­”жЎ€',
        # f'йЂ‰йЎ№\s?([{options}])\s?ж­ЈзЎ®',
        # f'ж‰Ђд»Ґз­”\s?([{options}])',
        # f'ж‰Ђд»Ґ\s?([{options}][.гЂ‚$]?$)',
        # f'ж‰Ђжњ‰\s?([{options}][.гЂ‚$]?$)',
        # f'[\sпјЊпјљ:,]([{options}])[гЂ‚пјЊ,\.]?$',
        # f'[\sпјЊ,пјљ:][ж•…еЌі]([{options}])[гЂ‚\.]?$',
        # f'[\sпјЊ,пјљ:]е› ж­¤([{options}])[гЂ‚\.]?$',
        # f'[жЇдёєгЂ‚]\s?([{options}])[гЂ‚\.]?$',
        # f'е› ж­¤\s?([{options}])[гЂ‚\.]?$',
        # f'жѕз„¶\s?([{options}])[гЂ‚\.]?$',
        # f'з­”жЎ€жЇ\s?(\S+)(?:гЂ‚|$)',
        # f'з­”жЎ€еє”иЇҐжЇ\s?(\S+)(?:гЂ‚|$)',
        # f'з­”жЎ€дёє\s?(\S+)(?:гЂ‚|$)',
        f'(?i)ANSWER\s*:\s*([{options}])',
        f'[Tt]he answer is:?\s+\(?([{options}])\)?',
        f'[Tt]he answer is:?\s+\(?\*?\*?([{options}])\*?\*?\)?',
        f'[Tt]he answer is option:?\s+\(?([{options}])\)?',
        f'[Tt]he correct answer is:?\s+\(?([{options}])\)?',
        f'[Tt]he correct answer is option:?\s+\(?([{options}])\)?',
        f'[Tt]he correct answer is:?.*?boxed{{([{options}])}}',
        f'[Tt]he correct option is:?.*?boxed{{([{options}])}}',
        f'[Tt]he correct answer option is:?.*?boxed{{([{options}])}}',
        f'[Tt]he answer to the question is:?\s+\(?([{options}])\)?',
        # f'^йЂ‰йЎ№\s?([{options}])',
        # f'^([{options}])\s?йЂ‰?йЎ№',
        f'(\s|^)[{options}][\sгЂ‚пјЊ,пјљ:\.$]',
        f'1.\s?(.*?)$',
        f'1.\s?([{options}])[.гЂ‚$]?$',
    ]
    cushion_patterns = [
        f'([{options}]):',
        f'([{options}])',
    ]
    # flake8: noqa
    # yapf: enable

    if cushion:
        patterns.extend(cushion_patterns)
    for pattern in patterns:
        text = text.strip()
        match = re.search(pattern, text, re.DOTALL)
        if match:
            if match.group(1) is not None and match.group(1) != '':
                outputs = match.group(1)
            else:
                outputs = match.group(0)
            for i in options:
                if i in outputs:
                    return i
    return ''


def get_letter_token_ids(tokenizer):
    letter_token_ids = {}
    for letter in ["A", "B", "C", "D"]:
        candidates = []
        for variant in [f" {letter}", letter]:
            ids = tokenizer(variant, add_special_tokens=False)["input_ids"]
            if len(ids) == 1:
                candidates.append(ids[0])
        letter_token_ids[letter] = list(dict.fromkeys(candidates))
    return letter_token_ids


def build_abcd_logits(logits_pos, letter_token_ids):
    abcd_logits = []
    for letter in ["A", "B", "C", "D"]:
        ids = letter_token_ids.get(letter, [])
        if not ids:
            raise ValueError(f"No single-token id found for letter '{letter}'")
        abcd_logits.append(torch.max(logits_pos[ids]))
    return torch.stack(abcd_logits)


def choose_preferred_letter_token_id(letter_token_ids, letter):
    ids = letter_token_ids.get(letter, [])
    return ids[0] if ids else None


def generate_responses(model, tokenizer, prompts, max_new_tokens=512):
    chat_texts = build_chat_texts(tokenizer, prompts)

    # OpenCompass uses left padding for generation
    tokenizer.padding_side = 'left'
    model_inputs = tokenizer(
        chat_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=False,  # chat template already added special tokens
    ).to(model.device)

    stop_words = ['<|im_end|>', '<|im_start|>']

    with torch.no_grad():
        outputs = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_ids = outputs[:, model_inputs["input_ids"].shape[1]:]

    responses = []
    for gen_ids in generated_ids:
        decoded = tokenizer.decode(gen_ids, skip_special_tokens=False)
        for stop in stop_words:
            decoded = decoded.split(stop)[0]
        answer = first_option_postprocess(decoded.strip())
        responses.append(answer)

    return responses


def test_accuracy(model, tokenizer, test_samples, batch_size=16):
    correct = 0
    total = len(test_samples)

    for start in tqdm(range(0, total, batch_size), desc="Testing accuracy", total=(total + batch_size - 1) // batch_size):
        if hasattr(test_samples, "select"):
            batch = test_samples.select(range(start, min(start + batch_size, total)))
            batch = [batch[k] for k in range(len(batch))]
        else:
            batch = test_samples[start : start + batch_size]

        prompts = [format_mmlu_question(ex) for ex in batch]
        responses = generate_responses(model, tokenizer, prompts)

        for example, response in zip(batch, responses):
            predicted = response
            correct_answer = ["A", "B", "C", "D"][example["answer"]]
            if predicted == correct_answer:
                correct += 1

    accuracy = correct / total
    return accuracy

def compute_expert_importance(model, tokenizer, dataset, loss_top_k='top_k_x2', generation_max_new_tokens=20):
    model.eval()
    num_layers = len(model.model.layers)

    expert_importance = [defaultdict(float) for _ in range(num_layers)]
    all_router_logits = defaultdict(list)
    letter_token_ids = get_letter_token_ids(tokenizer)
    stop_words = ['<|im_end|>', '<|im_start|>']

    for i, example in enumerate(tqdm(dataset, desc="Computing expert importance", total=len(dataset))):

        prompt = format_mmlu_question(example)
        correct_answer = ["A", "B", "C", "D"][example["answer"]]

        chat_text = build_chat_texts(tokenizer, [prompt])[0]
        inputs = tokenizer(
            [chat_text],
            return_tensors="pt",
            add_special_tokens=False,  # chat template already added special tokens
        ).to(model.device)
        prompt_ids = inputs["input_ids"]
        prompt_len = prompt_ids.shape[1]

        # 1) Generate up to N tokens and parse answer letter from generated text.
        with torch.no_grad():
            gen_outputs = model.generate(
                **inputs,
                max_new_tokens=generation_max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated_ids = gen_outputs[:, prompt_len:]
        if generated_ids.shape[1] == 0:
            print(f"[DEBUG] Example {i+1}: no generated tokens, skipping")
            continue

        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
        for stop in stop_words:
            generated_text = generated_text.split(stop)[0]
        generated_text = generated_text.strip()

        # Find the first generation token index where first_option_postprocess
        # can extract A/B/C/D from the decoded prefix.
        extracted_answer = ''
        extracted_gen_idx = None
        for step in range(generated_ids.shape[1]):
            prefix_text = tokenizer.decode(generated_ids[0, : step + 1], skip_special_tokens=False)
            for stop in stop_words:
                prefix_text = prefix_text.split(stop)[0]
            prefix_answer = first_option_postprocess(prefix_text.strip())
            if prefix_answer in ["A", "B", "C", "D"]:
                extracted_answer = prefix_answer
                extracted_gen_idx = step
                break

        print(f"[DEBUG] Example {i+1}: generated_text='{generated_text}'")
        #print(f"[DEBUG] Example {i+1}: extracted_answer='{extracted_answer}', gold='{correct_answer}'")

        if extracted_gen_idx is None:
            print(f"[DEBUG] Example {i+1}: no valid extracted option, skipping")
            continue

        print(f"[DEBUG] Example {i+1}: extracted_generation_step={extracted_gen_idx}")

        # 2) Build conditioning sequence up to the token BEFORE extracted answer.
        # If extracted token is first generated token (step 0), this is just prompt_ids.
        cond_gen_len = extracted_gen_idx
        if cond_gen_len > 0:
            conditioning_ids = torch.cat([prompt_ids, generated_ids[:, :cond_gen_len]], dim=1)
        else:
            conditioning_ids = prompt_ids

        pred_pos = conditioning_ids.shape[1] - 1

        hidden_states_with_grad = {}
        mlp_inputs = {}

        def make_backward_hook(layer_idx):
            def hook(module, grad_input, grad_output):
                if grad_output[0] is not None:
                    hidden_states_with_grad[layer_idx] = grad_output[0].clone()
            return hook

        def make_forward_hook(layer_idx):
            def hook(module, input, output):
                mlp_inputs[layer_idx] = input[0].detach().clone()
            return hook

        hooks = []
        for layer_idx in range(num_layers):
            backward_hook = model.model.layers[layer_idx].mlp.register_full_backward_hook(make_backward_hook(layer_idx))
            forward_hook = model.model.layers[layer_idx].mlp.register_forward_hook(make_forward_hook(layer_idx))
            hooks.append(backward_hook)
            hooks.append(forward_hook)

        outputs = model(input_ids=conditioning_ids, output_hidden_states=True)

        # Build A/B/C/D logits for the extracted generation step
        logits_pos = outputs.logits[0, pred_pos, :]
        abcd_logits = build_abcd_logits(logits_pos, letter_token_ids)
        probs_abcd = F.softmax(abcd_logits.float(), dim=-1)

        correct_idx = ["A", "B", "C", "D"].index(correct_answer)

        # Old mixed-scale loss (kept for reference):
        # p_correct = probs_abcd[correct_idx].clamp_min(1e-12)
        # loss_correct = -torch.log(p_correct)
        # loss_wrong = 1.0 - p_correct
        # beta = (1.0 - p_correct).detach()
        # loss = loss_correct + beta * loss_wrong

        # New loss: plain multiclass cross-entropy over A/B/C/D logits.
        target = torch.tensor([correct_idx], device=abcd_logits.device)
        loss = F.cross_entropy(abcd_logits.float().unsqueeze(0), target)
        p_correct = probs_abcd[correct_idx].clamp_min(1e-12)

        # Alternative (commented): probability from FULL vocabulary softmax,
        # then map to A/B/C/D via their token IDs (instead of 4-logit softmax only).
        #
        # probs_full = F.softmax(logits_pos.float(), dim=-1)  # over full vocab
        # p_correct_full = torch.max(probs_full[letter_token_ids[correct_answer]]).clamp_min(1e-12)
        # loss_correct_full = -torch.log(p_correct_full)
        # loss_wrong_full = 1.0 - p_correct_full
        # beta_full = (1.0 - p_correct_full).detach()
        # loss_full_vocab = loss_correct_full + beta_full * loss_wrong_full

        pred_idx = int(torch.argmax(probs_abcd).item())
        pred_letter = ["A", "B", "C", "D"][pred_idx]
        print(f"[DEBUG] Predicted={pred_letter}, extracted_answer={extracted_answer}, gold={correct_answer}")
        print(f"[DEBUG] Loss (CE over A/B/C/D logits): p_correct={p_correct.item():.6f}, loss={loss.item():.6f}")

        model.zero_grad()
        loss.backward()
        for hook in hooks:
            hook.remove()

        for layer_idx in range(num_layers):
            if layer_idx not in hidden_states_with_grad:
                print(f"[DEBUG] Layer {layer_idx}: no gradients, skipping")
                continue

            layer = model.model.layers[layer_idx]
            g = hidden_states_with_grad[layer_idx]
            hidden_flat = mlp_inputs[layer_idx]

            _, _, hidden_dim = hidden_flat.shape
            hidden_flat = hidden_flat.view(-1, hidden_dim)
            g_flat = g.view(-1, hidden_dim).to(hidden_flat.device)

            router_logits = layer.mlp.gate(hidden_flat)
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)

            # Determine effective top_k based on loss_top_k parameter
            base_top_k = layer.mlp.top_k
            num_experts = layer.mlp.num_experts
            if loss_top_k == 'top_k':
                effective_top_k = base_top_k
            elif loss_top_k == 'top_k_x2':
                effective_top_k = base_top_k * 2
            else:  # 'all'
                effective_top_k = num_experts

            weights, selected = torch.topk(routing_weights, effective_top_k, dim=-1)

            if layer.mlp.norm_topk_prob:
                weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-6)

            weights = weights.to(hidden_flat.device)
            selected = selected.to(hidden_flat.device)

            # Use exactly the same position as the extracted generation step predictor.
            grad_token_idx = pred_pos

            for k_idx in range(effective_top_k):
                expert_idx = selected[grad_token_idx, k_idx].item()
                w_i = weights[grad_token_idx, k_idx].item()

                with torch.no_grad():
                    h_i = layer.mlp.experts[expert_idx](hidden_flat[grad_token_idx : grad_token_idx + 1]).to(hidden_flat.device)

                g_token = g_flat[grad_token_idx]
                importance = -torch.dot(g_token, (w_i * h_i).squeeze(0)).item()
                expert_importance[layer_idx][expert_idx] += importance

            all_router_logits[layer_idx].append(router_logits[grad_token_idx])

        model.zero_grad()

    return expert_importance, all_router_logits

def apply_expert_bias(model, expert_importance, all_router_logits):
#     print(f"[DEBUG] Applying bias to experts")
    num_layers = len(model.model.layers)

    for layer_idx in range(num_layers):
#         print(f"[DEBUG] Processing layer {layer_idx}")
        if layer_idx >= BIAS_START_LAYER:
            continue
        layer = model.model.layers[layer_idx]

        if len(expert_importance[layer_idx]) == 0:
#             print(f"[DEBUG] Layer {layer_idx}: no expert-importance data, skipping")
            continue

        importances = torch.zeros(layer.mlp.num_experts)
        for expert_idx, imp in expert_importance[layer_idx].items():
            importances[expert_idx] = imp
#
#         print(f"[DEBUG] Layer {layer_idx}: raw importances - min={importances.min():.4f}, max={importances.max():.4f}, mean={importances.mean():.4f}")

        mean_imp = importances.mean()
        std_imp = importances.std()
#         print(f"[DEBUG] Layer {layer_idx}: mean={mean_imp:.4f}, std={std_imp:.4f}")

        if std_imp > 1e-8:
            importances = (importances - mean_imp) / std_imp
#             print(f"[DEBUG] Layer {layer_idx}: normalized importances - min={importances.min():.4f}, max={importances.max():.4f}")
        else:
            pass
#             print(f"[DEBUG] Layer {layer_idx}: std is too small ({std_imp:.6f}), skipping normalization")

#         print(f"[DEBUG] Computing eps = std(|logits|) / num_experts")
        logit_means_per_sample = []
        for router_logits in tqdm(all_router_logits[layer_idx]):
            logit_means_per_sample.append(router_logits.abs().std().item())

        logit_means_per_dataset = np.mean(logit_means_per_sample)

        with torch.no_grad():
            if layer.mlp.gate.bias is None:
                num_experts = layer.mlp.num_experts
                layer.mlp.gate.bias = nn.Parameter(torch.zeros(num_experts, dtype=layer.mlp.gate.weight.dtype, device=layer.mlp.gate.weight.device))
#                 print(f"[DEBUG] Layer {layer_idx}: created new bias parameter (size={num_experts})")

            top_k = model.model.layers[0].mlp.top_k
            eps = top_k * logit_means_per_dataset / layer.mlp.num_experts
            print(f"[DEBUG] Layer {layer_idx}: eps = {eps}")

            old_bias = layer.mlp.gate.bias.clone()
            layer.mlp.gate.bias += eps * importances.to(dtype=layer.mlp.gate.bias.dtype, device=layer.mlp.gate.bias.device)
            bias_change = torch.norm(layer.mlp.gate.bias - old_bias).item()
#             print(f"[DEBUG] Layer {layer_idx}: bias updated (norm change: {bias_change:.6f})")

        # Count how many experts get positive/negative/zero bias on this layer
        num_pos = int((importances > 0).sum().item())
        num_neg = int((importances < 0).sum().item())
        num_zero = int((importances == 0).sum().item())
        print(f"[DEBUG] Layer {layer_idx}: bias + for {num_pos}, bias - for {num_neg}, bias 0 for {num_zero}")

        # Top-5 and bottom-5 experts by normalized importance
        sorted_indices = torch.argsort(importances, descending=True)
        print(f"[DEBUG] Layer {layer_idx}: top-5 experts:")
        for rank in range(min(5, len(importances))):
            idx = sorted_indices[rank].item()
            print(f"Expert {idx}: importance = {importances[idx]:.4f}")

        print(f"[DEBUG] Layer {layer_idx}: bottom-5 experts:")
        for rank in range(1, min(5, len(importances)) + 1):
            idx = sorted_indices[-rank].item()
            print(f"Expert {idx}: importance = {importances[idx]:.4f}")
#
#     print(f"[DEBUG] Bias application completed for all layers")
    return model


def apply_expert_bias_v4(model, expert_importance, all_router_logits):
#     print(f"[DEBUG] Applying bias using v4 formula (Max-Abs norm + std-based eps)")
    num_layers = len(model.model.layers)
    SENSITIVITY = 1.0

    for layer_idx in range(num_layers):
        if layer_idx < BIAS_START_LAYER:
            continue
        layer = model.model.layers[layer_idx]
        num_experts = layer.mlp.num_experts
        top_k = getattr(model.config, "num_experts_per_tok", 4)

        if len(expert_importance[layer_idx]) == 0:
            continue

        importances = torch.zeros(num_experts)
        for expert_idx, imp in expert_importance[layer_idx].items():
            importances[expert_idx] = imp

        # Max-Abs normalization
        mean_imp = importances.mean()
        importances = importances - mean_imp
        max_abs = importances.abs().max()
        if max_abs > 1e-8:
            importances = importances / max_abs

        # Calculate logits
        all_logits_for_layer = torch.stack(all_router_logits[layer_idx])
        mean_router_logits = all_logits_for_layer.mean(dim=0)
        logit_std = mean_router_logits.std().item()

        with torch.no_grad():
            if layer.mlp.gate.bias is None:
                layer.mlp.gate.bias = nn.Parameter(torch.zeros(num_experts, dtype=layer.mlp.gate.weight.dtype, device=layer.mlp.gate.weight.device))

            # Ideal EPS formula
            eps = (logit_std * SENSITIVITY) / (num_experts / top_k)

            layer.mlp.gate.bias += eps * importances.to(dtype=layer.mlp.gate.bias.dtype, device=layer.mlp.gate.bias.device)
#
#     print(f"[DEBUG] Bias application completed for all layers")
    return model


def apply_expert_bias_v5(model, expert_importance, all_router_logits):
#     print(f"[DEBUG] Applying bias using v5 formula (Max-Abs norm + Top-K Logit Gap)")
    num_layers = len(model.model.layers)

    for layer_idx in range(num_layers):
        if layer_idx < BIAS_START_LAYER:
            continue
        layer = model.model.layers[layer_idx]
        num_experts = layer.mlp.num_experts
        top_k = getattr(model.config, "num_experts_per_tok", 4)

        if len(expert_importance[layer_idx]) == 0:
            continue

        importances = torch.zeros(num_experts)
        for expert_idx, imp in expert_importance[layer_idx].items():
            importances[expert_idx] = imp

        # Max-Abs normalization
        mean_imp = importances.mean()
        importances = importances - mean_imp
        max_abs = importances.abs().max()
        if max_abs > 1e-8:
            importances = importances / max_abs

        # Calculate adaptive EPS based on Logit Gap
        all_logits_for_layer = torch.stack(all_router_logits[layer_idx])
        mean_router_logits = all_logits_for_layer.mean(dim=0)

        sorted_logits, _ = torch.sort(mean_router_logits, descending=True)
        logit_gap = (sorted_logits[0] - sorted_logits[top_k]).item()
        eps = max(logit_gap, 1e-4)

        with torch.no_grad():
            if layer.mlp.gate.bias is None:
                layer.mlp.gate.bias = nn.Parameter(torch.zeros(num_experts, dtype=layer.mlp.gate.weight.dtype, device=layer.mlp.gate.weight.device))

            layer.mlp.gate.bias += eps * importances.to(dtype=layer.mlp.gate.bias.dtype, device=layer.mlp.gate.bias.device)
#
#     print(f"[DEBUG] Bias application completed for all layers")
    return model


def apply_expert_bias_v6(model, expert_importance, all_router_logits):
    print("[DEBUG] Applying bias by formula v6 (gap-normalized sparse bias)")
    num_layers = len(model.model.layers)
    layer_stats = []
    positive_bias_layers_applied = []
    negative_bias_layers_applied = []

    # Pass 1: collect per-layer stats needed for global quantile thresholds.
    for layer_idx in range(num_layers):
        if layer_idx >= BIAS_START_LAYER:
            continue

        layer = model.model.layers[layer_idx]
        num_experts = layer.mlp.num_experts

        if len(all_router_logits[layer_idx]) == 0:
            print(f"[DEBUG][Layer {layer_idx}] skipped: no router logits collected")
            continue

        all_logits_tensor = torch.stack(all_router_logits[layer_idx]).float()
        # flat_logits = all_logits_tensor.reshape(-1)
        # act_min = flat_logits.min().item()
        # act_max = flat_logits.max().item()
        # act_range = act_max - act_min
        # act_mean = flat_logits.mean().item()
        # act_std = flat_logits.std(unbiased=False).item()
        # act_p05 = torch.quantile(flat_logits, 0.05).item()
        # act_p95 = torch.quantile(flat_logits, 0.95).item()

        sorted_logits, _ = torch.sort(all_logits_tensor, dim=1, descending=True)
        mean_gap_top1_top2 = (sorted_logits[:, 0] - sorted_logits[:, 1]).mean().item()
        top_k = int(getattr(layer.mlp, "top_k", getattr(model.config, "num_experts_per_tok", 4)))
        if top_k < num_experts:
            mean_gap_topk_topk1 = (sorted_logits[:, top_k - 1] - sorted_logits[:, top_k]).mean().item()
        else:
            mean_gap_topk_topk1 = float("nan")

        importances = torch.zeros(num_experts)
        for expert_idx, imp in expert_importance[layer_idx].items():
            importances[expert_idx] = imp

        mean_imp = importances.mean()
        importances = importances - mean_imp
        max_abs = importances.abs().max()
        if max_abs > 1e-8:
            importances = importances / max_abs

        sorted_indices = torch.argsort(importances, descending=True)
        top_3 = [(idx.item(), importances[idx].item()) for idx in sorted_indices[:3]]
        bottom_3 = [(idx.item(), importances[idx].item()) for idx in sorted_indices[-3:]]

        g1 = mean_gap_top1_top2
        gk = mean_gap_topk_topk1
        eps = (g1 * gk) / max(g1 + gk, 1e-8)
        safe_gap = max(gk, 1e-8)

        pos_indices = torch.where(importances > 0)[0]
        neg_indices = torch.where(importances < 0)[0]

        selected_pos = torch.empty(0, dtype=torch.long, device=importances.device)
        selected_neg = torch.empty(0, dtype=torch.long, device=importances.device)
        pos_layer_mean_threshold = 0.0
        neg_layer_mean_abs = 0.0
        neg_layer_std_abs = 0.0
        neg_layer_threshold = 0.0
        pos_top3_mean = 0.0
        neg_top3_abs_mean = 0.0

        # Old fixed top-3 expert-selection logic kept for reference:
        # if pos_indices.numel() > 0:
        #     pos_vals = importances[pos_indices]
        #     k_pos = min(NUM_EXP_POS_BIAS_APPLY, pos_indices.numel())
        #     pos_order = torch.argsort(pos_vals, descending=True)
        #     selected_pos = pos_indices[pos_order[:k_pos]]
        #     pos_top3_mean = importances[selected_pos].mean().item()
        #
        # if neg_indices.numel() > 0:
        #     neg_abs_vals = importances[neg_indices].abs()
        #     k_neg = min(NUM_EXP_NEG_BIAS_APPLY, neg_indices.numel())
        #     neg_order = torch.argsort(neg_abs_vals, descending=True)
        #     selected_neg = neg_indices[neg_order[:k_neg]]
        #     neg_top3_abs_mean = importances[selected_neg].abs().mean().item()

        if pos_indices.numel() > 0:
            pos_vals = importances[pos_indices]
            pos_layer_mean_threshold = pos_vals.mean().item()
            selected_pos = pos_indices[pos_vals >= pos_layer_mean_threshold]

        if neg_indices.numel() > 0:
            neg_abs_vals = importances[neg_indices].abs()
            neg_layer_mean_abs = neg_abs_vals.mean().item()
            neg_layer_std_abs = neg_abs_vals.std(unbiased=False).item()
            neg_layer_threshold = neg_layer_mean_abs + neg_layer_std_abs
            selected_neg = neg_indices[neg_abs_vals >= neg_layer_threshold]

        pos_mean = importances[selected_pos].mean().item() if selected_pos.numel() > 0 else 0.0
        neg_mean_abs = importances[selected_neg].abs().mean().item() if selected_neg.numel() > 0 else 0.0
        pos_auto_score = (eps * pos_mean) / safe_gap if selected_pos.numel() > 0 else 0.0
        neg_auto_score = (eps * neg_mean_abs) / safe_gap if selected_neg.numel() > 0 else 0.0

        layer_stats.append({
            "layer_idx": layer_idx,
            "layer": layer,
            "num_experts": num_experts,
            "top_k": top_k,
            "g1": g1,
            "gk": gk,
            "eps": eps,
            "safe_gap": safe_gap,
            "importances": importances,
            "top_3": top_3,
            "bottom_3": bottom_3,
            "selected_pos": selected_pos,
            "selected_neg": selected_neg,
            "pos_layer_mean_threshold": pos_layer_mean_threshold,
            "neg_layer_mean_abs": neg_layer_mean_abs,
            "neg_layer_std_abs": neg_layer_std_abs,
            "neg_layer_threshold": neg_layer_threshold,
            "pos_mean": pos_mean,
            "neg_mean_abs": neg_mean_abs,
            "pos_auto_score": pos_auto_score,
            "neg_auto_score": neg_auto_score,
        })

    pos_scores_all_layers = [s["pos_auto_score"] for s in layer_stats]
    neg_scores_all_layers = [s["neg_auto_score"] for s in layer_stats]

    pos_otsu_threshold = otsu_threshold(pos_scores_all_layers)
    neg_otsu_threshold_stage1 = otsu_threshold(neg_scores_all_layers)
    neg_layers_after_stage1 = [s["layer_idx"] for s in layer_stats if s["neg_auto_score"] >= neg_otsu_threshold_stage1]
    neg_high_cluster_scores = [score for score in neg_scores_all_layers if score >= neg_otsu_threshold_stage1]
    neg_otsu_threshold_stage2 = otsu_threshold(neg_high_cluster_scores) if neg_high_cluster_scores else neg_otsu_threshold_stage1
    neg_layers_after_stage2 = [s["layer_idx"] for s in layer_stats if s["neg_auto_score"] >= neg_otsu_threshold_stage2]

    # Old quantile layer-selection logic kept for reference:
    # pos_quantile_threshold = (
    #     float(torch.quantile(torch.tensor(pos_scores_all_layers, dtype=torch.float32), POSITIVE_BIAS_AUTO_QUANTILE).item())
    #     if pos_scores_all_layers else 0.0
    # )
    # neg_quantile_threshold = (
    #     float(torch.quantile(torch.tensor(neg_scores_all_layers, dtype=torch.float32), NEGATIVE_BIAS_AUTO_QUANTILE).item())
    #     if neg_scores_all_layers else 0.0
    # )

    print(
        f"[DEBUG][v6] auto thresholds (Otsu): "
        f"pos_threshold={pos_otsu_threshold:.8f} from scores={pos_scores_all_layers}"
    )
    print(
        f"[DEBUG][v6] negative thresholds (double Otsu): "
        f"stage1={neg_otsu_threshold_stage1:.8f} from scores={neg_scores_all_layers}; "
        f"stage2={neg_otsu_threshold_stage2:.8f} from high_cluster_scores={neg_high_cluster_scores}"
    )
    print(
        f"[DEBUG][v6] negative layers after stage1 Otsu: "
        f"count={len(neg_layers_after_stage1)} layers={neg_layers_after_stage1}"
    )
    print(
        f"[DEBUG][v6] negative layers after stage2 Otsu: "
        f"count={len(neg_layers_after_stage2)} layers={neg_layers_after_stage2}"
    )

    # Pass 2: decide per layer and apply sparse bias updates.
    for stats in layer_stats:
        layer_idx = stats["layer_idx"]
        layer = stats["layer"]
        num_experts = stats["num_experts"]
        top_k = stats["top_k"]
        g1 = stats["g1"]
        gk = stats["gk"]
        eps = stats["eps"]
        importances = stats["importances"]
        top_3 = stats["top_3"]
        bottom_3 = stats["bottom_3"]
        selected_pos = stats["selected_pos"]
        selected_neg = stats["selected_neg"]
        pos_layer_mean_threshold = stats["pos_layer_mean_threshold"]
        neg_layer_mean_abs = stats["neg_layer_mean_abs"]
        neg_layer_std_abs = stats["neg_layer_std_abs"]
        neg_layer_threshold = stats["neg_layer_threshold"]
        pos_mean = stats["pos_mean"]
        neg_mean_abs = stats["neg_mean_abs"]
        pos_auto_score = stats["pos_auto_score"]
        neg_auto_score = stats["neg_auto_score"]

        pos_layer_allowed = (
            layer_idx in POSITIVE_BIAS_LAYERS
            if POSITIVE_BIAS_LAYERS is not None
            else pos_auto_score >= pos_otsu_threshold
        )
        neg_layer_allowed = (
            layer_idx in NEGATIVE_BIAS_LAYERS
            if NEGATIVE_BIAS_LAYERS is not None
            else neg_auto_score >= neg_otsu_threshold_stage2
        )

        selected_pos_final = selected_pos if pos_layer_allowed else torch.empty(0, dtype=torch.long, device=importances.device)
        selected_neg_final = selected_neg if neg_layer_allowed else torch.empty(0, dtype=torch.long, device=importances.device)

        if selected_pos_final.numel() > 0:
            positive_bias_layers_applied.append(layer_idx)
        if selected_neg_final.numel() > 0:
            negative_bias_layers_applied.append(layer_idx)

        selected_indices = torch.cat([selected_pos_final, selected_neg_final]).unique()
        sparse_importances = torch.zeros_like(importances)
        sparse_importances[selected_indices] = importances[selected_indices]
        delta_bias = eps * sparse_importances

        with torch.no_grad():
            if layer.mlp.gate.bias is None:
                layer.mlp.gate.bias = nn.Parameter(torch.zeros(num_experts, dtype=layer.mlp.gate.weight.dtype, device=layer.mlp.gate.weight.device))
            layer.mlp.gate.bias += delta_bias.to(dtype=layer.mlp.gate.bias.dtype, device=layer.mlp.gate.bias.device)

        pos_updates = [(idx.item(), delta_bias[idx].item()) for idx in selected_pos_final]
        neg_updates = [(idx.item(), delta_bias[idx].item()) for idx in selected_neg_final]

        print(f"[DEBUG][Layer {layer_idx}] gaps: mean_gap(top1-top2)={g1:.6f} mean_gap(top{top_k}-top{top_k + 1})={gk:.6f}")
        print(f"[DEBUG][Layer {layer_idx}] bias counts: pos={selected_pos_final.numel()} neg={selected_neg_final.numel()} top3={top_3} bottom3={bottom_3}")
        print(
            f"[DEBUG][Layer {layer_idx}] expert selection thresholds: "
            f"pos: importance >= mean(positive) = {pos_layer_mean_threshold:.8f}; "
            f"neg: abs(importance) >= mean(abs(negative)) + std(abs(negative)) = "
            f"{neg_layer_mean_abs:.8f} + {neg_layer_std_abs:.8f} = {neg_layer_threshold:.8f}"
        )
        print(
            f"[DEBUG][Layer {layer_idx}] auto-score formula: "
            f"pos_score=(eps * pos_mean) / gap = ({eps:.8f} * {pos_mean:.8f}) / {max(gk, 1e-8):.8f} = {pos_auto_score:.8f}; "
            f"neg_score=(eps * neg_mean_abs) / gap = ({eps:.8f} * {neg_mean_abs:.8f}) / {max(gk, 1e-8):.8f} = {neg_auto_score:.8f}"
        )
        print(f"[DEBUG][Layer {layer_idx}] eps={eps:.8f}")
        print(
            f"[DEBUG][Layer {layer_idx}] decisions: "
            f"pos_allowed={pos_layer_allowed} "
            f"({'manual' if POSITIVE_BIAS_LAYERS is not None else f'auto_otsu>={pos_otsu_threshold:.8f}'}) "
            f"neg_allowed={neg_layer_allowed} "
            f"({'manual' if NEGATIVE_BIAS_LAYERS is not None else f'auto_double_otsu>={neg_otsu_threshold_stage2:.8f}'})"
        )
        print(f"[DEBUG][Layer {layer_idx}] sparse experts for bias update: pos={pos_updates} neg={neg_updates}")

    print(
        f"[DEBUG][v6] final positive bias layers: "
        f"count={len(positive_bias_layers_applied)} layers={positive_bias_layers_applied}"
    )
    print(
        f"[DEBUG][v6] final negative bias layers: "
        f"count={len(negative_bias_layers_applied)} layers={negative_bias_layers_applied}"
    )

    return model

def apply_expert_bias_v7(model, expert_importance, all_router_logits):
#     print(f"[DEBUG] Applying bias using v7 formula (Max-Abs norm + Robust Quantile Step)")
    num_layers = len(model.model.layers)

    for layer_idx in range(num_layers):
        if layer_idx < BIAS_START_LAYER:
            continue
        layer = model.model.layers[layer_idx]
        num_experts = layer.mlp.num_experts

        if len(expert_importance[layer_idx]) == 0:
            continue

        importances = torch.zeros(num_experts)
        for expert_idx, imp in expert_importance[layer_idx].items():
            importances[expert_idx] = imp

        # Max-Abs normalization
        mean_imp = importances.mean()
        importances = importances - mean_imp
        max_abs = importances.abs().max()
        if max_abs > 1e-8:
            importances = importances / max_abs

        # Calculate EPS based on quantiles (robust to outliers/Kurtosis)
        all_logits_tensor = torch.stack(all_router_logits[layer_idx]).float()
        q75 = torch.quantile(all_logits_tensor, 0.75).item()
        q25 = torch.quantile(all_logits_tensor, 0.25).item()
        iqr = q75 - q25

        eps = iqr / num_experts

        with torch.no_grad():
            if layer.mlp.gate.bias is None:
                layer.mlp.gate.bias = nn.Parameter(torch.zeros(num_experts, dtype=layer.mlp.gate.weight.dtype, device=layer.mlp.gate.weight.device))

            layer.mlp.gate.bias += eps * importances.to(dtype=layer.mlp.gate.bias.dtype, device=layer.mlp.gate.bias.device)
#
#     print(f"[DEBUG] Bias application completed for all layers")
    return model


def apply_expert_bias_default(model, expert_importance, all_router_logits, eps_value):
    num_layers = len(model.model.layers)

    for layer_idx in range(num_layers):
        if layer_idx < BIAS_START_LAYER:
            continue
        layer = model.model.layers[layer_idx]
        num_experts = layer.mlp.num_experts

        if len(expert_importance[layer_idx]) == 0:
            continue

        importances = torch.zeros(num_experts)
        for expert_idx, imp in expert_importance[layer_idx].items():
            importances[expert_idx] = imp

        # Max-Abs normalization
        mean_imp = importances.mean()
        importances = importances - mean_imp
        max_abs = importances.abs().max()
        if max_abs > 1e-8:
            importances = importances / max_abs

        # Use fixed eps value
        eps = eps_value

        with torch.no_grad():
            layer.mlp.gate.bias = nn.Parameter(torch.zeros(num_experts, dtype=layer.mlp.gate.weight.dtype, device=layer.mlp.gate.weight.device))
            layer.mlp.gate.bias.data = (eps * importances).to(dtype=layer.mlp.gate.weight.dtype, device=layer.mlp.gate.bias.device)

    return model


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        "/mnt/storage00/usr/ishukin/models/qwen-moe-chat",
        padding_side='left',
        truncation_side='left',
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        "/mnt/storage00/usr/ishukin/models/qwen-moe-chat",
        torch_dtype=torch.float16,
        device_map="auto",
        # num_experts_per_tok=4,
    )

    return model, tokenizer


def run_calibration_method(model, expert_importance, all_router_logits, args):
    if args.method == 'baseline':
        return model
    if args.method == 'original':
        return apply_expert_bias(model, expert_importance, all_router_logits)
    if args.method == 'v4':
        return apply_expert_bias_v4(model, expert_importance, all_router_logits)
    if args.method == 'v5':
        return apply_expert_bias_v5(model, expert_importance, all_router_logits)
    if args.method == 'v6':
        return apply_expert_bias_v6(model, expert_importance, all_router_logits)
    if args.method == 'v7':
        return apply_expert_bias_v7(model, expert_importance, all_router_logits)
    if args.method == 'default':
        if args.eps_value is None:
            raise ValueError("--eps_value must be specified when using --method default with loss_top_k != 'top_k'")
        return apply_expert_bias_default(model, expert_importance, all_router_logits, args.eps_value)
    raise ValueError(f"Unsupported method: {args.method}")


def evaluate_named_datasets(model, tokenizer, named_datasets, prefix):
    for dataset_name, dataset in named_datasets:
        accuracy = test_accuracy(model, tokenizer, dataset)
        print(f"{prefix} {dataset_name}:", accuracy)


def cleanup_cuda_objects(*objects):
    for obj in objects:
        if obj is None:
            continue
        if hasattr(obj, "zero_grad"):
            try:
                obj.zero_grad(set_to_none=True)
            except TypeError:
                obj.zero_grad()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def otsu_threshold(values):
    values = sorted(float(v) for v in values)
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]

    candidates = []
    for i in range(len(values) - 1):
        if values[i] != values[i + 1]:
            candidates.append(0.5 * (values[i] + values[i + 1]))

    if not candidates:
        return values[0]

    n = len(values)
    best_threshold = candidates[0]
    best_between_class_var = -1.0

    for threshold in candidates:
        left = [x for x in values if x < threshold]
        right = [x for x in values if x >= threshold]
        if not left or not right:
            continue

        w0 = len(left) / n
        w1 = len(right) / n
        mu0 = sum(left) / len(left)
        mu1 = sum(right) / len(right)
        between_class_var = w0 * w1 * (mu0 - mu1) ** 2

        if between_class_var > best_between_class_var:
            best_between_class_var = between_class_var
            best_threshold = threshold

    return best_threshold


def get_domain_configs():
    return {
        "math": {
            "calibration_dataset": test_samples_math,
            "eval_datasets": [
                ("college_mathematics", test_samples_math),
                ("abstract_algebra", test_samples_alg),
                ("elementary_mathematics", test_samples_elem_math),
                ("high_school_mathematics", test_samples_high_school_math),
            ],
        },
        "physics": {
            "calibration_dataset": test_samples_college_physics,
            "eval_datasets": [
                ("college_physics", test_samples_college_physics),
                ("conceptual_physics", test_samples_conceptual_physics),
                ("high_school_physics", test_samples_high_school_physics),
            ],
        },
        "history": {
            "calibration_dataset": test_samples_high_school_world_history_part,
            "eval_datasets": [
                ("high_school_world_history", test_samples_high_school_world_history),
                ("high_school_european_history", test_samples_high_school_european_history),
                ("high_school_us_history", test_samples_high_school_us_history),
                ("prehistory", test_samples_prehistory),
            ],
        },
        "biology": {
            "calibration_dataset": test_samples_college_biology_part,
            "eval_datasets": [
                ("college_biology", test_samples_college_biology),
                ("high_school_biology", test_samples_high_school_biology),
            ],
        },
        "chemistry": {
            "calibration_dataset": test_samples_college_chemistry_part,
            "eval_datasets": [
                ("college_chemistry", test_samples_college_chemistry),
                ("high_school_chemistry", test_samples_high_school_chemistry),
            ],
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='WES Model Calibration')
    parser.add_argument('--method', type=str, default='v6',
                       choices=['baseline', 'original', 'v4', 'v5', 'v6', 'v7', 'default'],
                       help='Calibration method to use')
    parser.add_argument('--mode', type=str, default='single',
                       choices=['single', 'all'],
                       help='Run a single-domain experiment (default) or all configured domains sequentially')
    parser.add_argument('--domain', type=str, default='math',
                       choices=['math', 'physics', 'history', 'biology', 'chemistry'],
                       help='Domain to run in --mode single')
    parser.add_argument('--loss_top_k', type=str, default='top_k',
                       choices=['top_k', 'top_k_x2', 'all'],
                       help='Loss calculation mode: top_k (default), top_k_x2 (top_k*2), or all (num_experts)')
    parser.add_argument('--eps_value', type=float, default=None,
                       help='Fixed epsilon value for default method (e.g., 0.005, 0.01, 0.05, 0.1, 0.2, 0.3)')
    args = parser.parse_args()


    # Check CUDA availability
#     print(f"[DEBUG] CUDA check: available = {torch.cuda.is_available()}")
    if torch.cuda.is_available():
#         print(f"[DEBUG] Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            pass
#             print(f"[DEBUG] GPU {i}: {torch.cuda.get_device_name(i)}")
#             print(f"[DEBUG] GPU {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")

    domain_configs = get_domain_configs()

    if args.mode == 'all' and args.method != 'baseline':
        for domain_name in ["math", "physics", "history", "biology", "chemistry"]:
            print(f"\n[DEBUG] ===== DOMAIN: {domain_name} =====")
            cmd = [
                sys.executable,
                os.path.abspath(__file__),
                "--method", args.method,
                "--mode", "single",
                "--domain", domain_name,
                "--loss_top_k", args.loss_top_k,
            ]
            if args.eps_value is not None:
                cmd.extend(["--eps_value", str(args.eps_value)])

            completed = subprocess.run(cmd)
            if completed.returncode != 0:
                raise SystemExit(completed.returncode)

        raise SystemExit(0)

    # OpenCompass uses padding_side='left' and truncation_side='left' by default
    model, tokenizer = load_model_and_tokenizer()
#
#     print(f"[DEBUG] Model loaded on device: {model.device}")
#     print(f"[DEBUG] Model dtype: {model.dtype}")


    # Only test before tuning for baseline mode
    if args.method == 'baseline':
        print("\n[DEBUG] ===== TESTING BEFORE TUNING =====")
        #accuracy = test_accuracy(model, tokenizer, test_samples_math)
        #print("BEFORE college_mathematics:", accuracy)

        #accuracy = test_accuracy(model, tokenizer, test_samples_elem_math)
        #print("BEFORE elementary_mathematics:", accuracy)

        #accuracy = test_accuracy(model, tokenizer, test_samples_high_school_math)
        #print("BEFORE high_school_mathematics:", accuracy)

        #accuracy = test_accuracy(model, tokenizer, test_samples_alg)
        #print("BEFORE abstract_algebra:", accuracy)

        #accuracy = test_accuracy(model, tokenizer, test_samples_high_school_physics)
        #print("BEFORE high_school_physics:", accuracy)

        #accuracy = test_accuracy(model, tokenizer, test_samples_conceptual_physics)
        #print("BEFORE conceptual_physics:", accuracy)

        #accuracy = test_accuracy(model, tokenizer, test_samples_college_physics)
        #print("BEFORE college_physics:", accuracy)

        #accuracy = test_accuracy(model, tokenizer, test_samples_high_school_biology)
        #print("BEFORE high_school_biology:", accuracy)

        #accuracy = test_accuracy(model, tokenizer, test_samples_college_biology)
        #print("BEFORE college_biology:", accuracy)

        accuracy = test_accuracy(model, tokenizer, test_samples_high_school_chemistry)
        print("BEFORE high_school_chemistry:", accuracy)

        accuracy = test_accuracy(model, tokenizer, test_samples_college_chemistry)
        print("BEFORE college_chemistry:", accuracy)

        #accuracy = test_accuracy(model, tokenizer, test_samples_high_school_us_history)
        #print("BEFORE high_school_us_history:", accuracy)

        #accuracy = test_accuracy(model, tokenizer, test_samples_high_school_european_history)
        #print("BEFORE high_school_european_history:", accuracy)

        #accuracy = test_accuracy(model, tokenizer, test_samples_high_school_world_history)
        #print("BEFORE high_school_world_history:", accuracy)

        #accuracy = test_accuracy(model, tokenizer, test_samples_prehistory)
        #print("BEFORE prehistory:", accuracy)

        #accuracy = test_accuracy(model, tokenizer, test_samples_college_med)
        #print("BEFORE college_medicine:", accuracy)

        #accuracy = test_accuracy(model, tokenizer, test_samples_prof_med)
        #print("BEFORE professional_medicine:", accuracy)

        #accuracy = test_accuracy(model, tokenizer, test_samples_clin_know)
        #print("BEFORE clinical_knowledge:", accuracy)

        #accuracy = test_accuracy(model, tokenizer, test_samples_all_part)
        #print("BEFORE all_part:", accuracy)

        #accuracy = test_accuracy(model, tokenizer, test_samples_all)
        #print("BEFORE all:", accuracy)

        print()


#     print("\n[DEBUG] ===== COMPUTING EXPERT IMPORTANCE =====")
    selected_domain = domain_configs[args.domain]
    calibration_dataset = selected_domain["calibration_dataset"]
    eval_datasets = selected_domain["eval_datasets"]

    expert_importance, mlp_inputs = compute_expert_importance(model, tokenizer, calibration_dataset, args.loss_top_k)

#     print("\n[DEBUG] ===== APPLYING EXPERT BIAS =====")

    # If method is 'default' and loss_top_k is 'top_k', run for all eps values
    if args.method == 'default' and args.loss_top_k == 'top_k':
        eps_values = [0.005, 0.01, 0.05, 0.1, 0.2, 0.3]

        # Compute expert importance once
        print("Computing expert importance (once for all eps values)...")
        expert_importance_computed, mlp_inputs_computed = compute_expert_importance(model, tokenizer, test_samples_math, args.loss_top_k)

        for eps_val in eps_values:
            print(f"\n{'='*60}")
            print(f"Running with eps_value = {eps_val}")
            print(f"{'='*60}")

            # Apply bias with current eps value (sets bias directly, not adding)
            model = apply_expert_bias_default(model, expert_importance_computed, mlp_inputs_computed, eps_val)

            # Cleanup
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()

            # Test accuracy
            accuracy = test_accuracy(model, tokenizer, test_samples_math)
            print(f"AFTER college_mathematics (eps={eps_val}):", accuracy)

            accuracy = test_accuracy(model, tokenizer, test_samples_alg)
            print(f"AFTER abstract_algebra (eps={eps_val}):", accuracy)

            accuracy = test_accuracy(model, tokenizer, test_samples_all_part)
            print(f"AFTER all_part (eps={eps_val}):", accuracy)
    else:
        # Original single-run logic
        model = run_calibration_method(model, expert_importance, mlp_inputs, args)

        # Cleanup to free memory before accuracy evaluation
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()

        # Only test after tuning if calibration was applied
        if args.method != 'baseline':
#         print("\n[DEBUG] ===== TESTING AFTER TUNING =====")
            evaluate_named_datasets(model, tokenizer, eval_datasets, "AFTER")

            if args.domain == "math":
                accuracy = test_accuracy(model, tokenizer, test_samples_all_part)
                print("AFTER all_part:", accuracy)

     #print("[DEBUG] ===== DONE =====")
