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


def compute_answer_loss(
    abcd_logits,
    correct_idx,
    loss_type,
    *,
    tau=0.0,
    lambda_ce=0.5,
    boundary_scale=1.0,
    boundary_weight_min=0.1,
    boundary_weight_max=1.0,
):
    abcd_logits = abcd_logits.float()
    target = torch.tensor([correct_idx], device=abcd_logits.device)
    ce_loss = F.cross_entropy(abcd_logits.unsqueeze(0), target)

    wrong_mask = torch.ones(abcd_logits.shape[0], dtype=torch.bool, device=abcd_logits.device)
    wrong_mask[correct_idx] = False
    gold_logit = abcd_logits[correct_idx]
    max_wrong_logit = torch.max(abcd_logits[wrong_mask])
    margin = gold_logit - max_wrong_logit
    wrong_logsumexp = torch.logsumexp(abcd_logits[wrong_mask], dim=0)
    soft_margin_loss = F.softplus(tau + wrong_logsumexp - gold_logit)
    boundary_weight = torch.exp(-torch.abs(margin.detach()) / boundary_scale)
    boundary_weight = torch.clamp(boundary_weight, boundary_weight_min, boundary_weight_max)
    boundary_weighted_margin_loss = boundary_weight * soft_margin_loss

    if loss_type == 'ce':
        loss = ce_loss
    elif loss_type == 'soft_margin':
        loss = soft_margin_loss
    elif loss_type == 'ce_margin_hybrid':
        loss = lambda_ce * ce_loss + (1.0 - lambda_ce) * soft_margin_loss
    elif loss_type == 'boundary_weighted_margin':
        loss = boundary_weighted_margin_loss
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    return loss, {
        "ce_loss": ce_loss.detach(),
        "soft_margin_loss": soft_margin_loss.detach(),
        "boundary_weighted_margin_loss": boundary_weighted_margin_loss.detach(),
        "boundary_weight": boundary_weight.detach(),
        "gold_logit": gold_logit.detach(),
        "max_wrong_logit": max_wrong_logit.detach(),
        "margin": margin.detach(),
    }


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


def iter_dataset_batches(test_samples, batch_size):
    total = len(test_samples)
    for start in range(0, total, batch_size):
        if hasattr(test_samples, "select"):
            batch = test_samples.select(range(start, min(start + batch_size, total)))
            batch = [batch[k] for k in range(len(batch))]
        else:
            batch = test_samples[start : start + batch_size]
        yield batch


def count_correct_generated_answers(model, tokenizer, test_samples, batch_size=16):
    correct = 0
    total = len(test_samples)

    for start in tqdm(range(0, total, batch_size), desc="Generate-validate accuracy", total=(total + batch_size - 1) // batch_size):
        if hasattr(test_samples, "select"):
            batch = test_samples.select(range(start, min(start + batch_size, total)))
            batch = [batch[k] for k in range(len(batch))]
        else:
            batch = test_samples[start : start + batch_size]

        prompts = [format_mmlu_question(ex) for ex in batch]
        responses = generate_responses(model, tokenizer, prompts)

        for example, response in zip(batch, responses):
            correct_answer = ["A", "B", "C", "D"][example["answer"]]
            if response == correct_answer:
                correct += 1

    return correct, total


def count_correct_next_token_answers(model, tokenizer, test_samples, batch_size=16):
    correct = 0
    total = len(test_samples)
    letter_token_ids = get_letter_token_ids(tokenizer)

    for batch in tqdm(iter_dataset_batches(test_samples, batch_size), desc="Next-token accuracy", total=(total + batch_size - 1) // batch_size):
        prompts = [format_mmlu_question(ex) for ex in batch]
        chat_texts = build_chat_texts(tokenizer, prompts)

        tokenizer.padding_side = 'left'
        inputs = tokenizer(
            chat_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)

        for row_idx, example in enumerate(batch):
            if "attention_mask" in inputs:
                token_positions = torch.nonzero(inputs["attention_mask"][row_idx], as_tuple=False).flatten()
                pred_pos = token_positions[-1].item()
            else:
                pred_pos = inputs["input_ids"].shape[1] - 1

            logits_pos = outputs.logits[row_idx, pred_pos, :]
            abcd_logits = build_abcd_logits(logits_pos, letter_token_ids)
            pred_idx = int(torch.argmax(abcd_logits.float()).item())
            if pred_idx == int(example["answer"]):
                correct += 1

    return correct, total


def count_correct_answers(model, tokenizer, test_samples, eval_mode='generation', batch_size=16):
    if eval_mode == 'generation':
        return count_correct_generated_answers(model, tokenizer, test_samples, batch_size=batch_size)
    if eval_mode == 'next_token_pred':
        return count_correct_next_token_answers(model, tokenizer, test_samples, batch_size=batch_size)
    raise ValueError(f"Unsupported eval_mode: {eval_mode}")


def test_accuracy(model, tokenizer, test_samples, batch_size=16, eval_mode='generation'):
    correct, total = count_correct_answers(model, tokenizer, test_samples, eval_mode=eval_mode, batch_size=batch_size)
    return correct / total


def take_validation_subset(dataset, validation_size):
    total = len(dataset)
    if validation_size <= 0 or validation_size >= total:
        return dataset
    start = total - validation_size
    indices = range(start, total)
    if hasattr(dataset, "select"):
        return dataset.select(indices)
    return dataset[start:total]


def add_gate_bias_delta(model, layer_idx, expert_idx, delta):
    layer = model.model.layers[layer_idx]
    num_experts = layer.mlp.num_experts
    with torch.no_grad():
        if layer.mlp.gate.bias is None:
            layer.mlp.gate.bias = nn.Parameter(torch.zeros(num_experts, dtype=layer.mlp.gate.weight.dtype, device=layer.mlp.gate.weight.device))
        layer.mlp.gate.bias[expert_idx] += torch.tensor(delta, dtype=layer.mlp.gate.bias.dtype, device=layer.mlp.gate.bias.device)


def clone_gate_bias(model, layer_idx):
    bias = model.model.layers[layer_idx].mlp.gate.bias
    return None if bias is None else bias.detach().clone()


def restore_gate_bias(model, layer_idx, bias_snapshot):
    layer = model.model.layers[layer_idx]
    with torch.no_grad():
        if bias_snapshot is None:
            layer.mlp.gate.bias = None
        else:
            restored = bias_snapshot.to(dtype=layer.mlp.gate.weight.dtype, device=layer.mlp.gate.weight.device)
            if layer.mlp.gate.bias is None:
                layer.mlp.gate.bias = nn.Parameter(restored.clone())
            else:
                layer.mlp.gate.bias.data.copy_(restored)


def parse_float_list(value):
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    items = []
    for raw in str(value).split(","):
        raw = raw.strip()
        if raw:
            items.append(float(raw))
    if not items:
        raise ValueError("Float list must contain at least one value")
    return items


def generate_validate_select_and_apply(model, tokenizer, calibration_dataset, candidates, args):
    validation_dataset = take_validation_subset(calibration_dataset, args.selection_validation_size)
    candidates = sorted(candidates, key=lambda item: abs(item["delta"]), reverse=True)
    if args.selection_max_candidates > 0:
        candidates = candidates[:args.selection_max_candidates]
    delta_multipliers = args.selection_delta_multipliers

    print(
        f"[DEBUG][GenerateValidate] start: validation_size={len(validation_dataset)} "
        f"candidate_count={len(candidates)} max_selected={args.selection_max_selected} "
        f"min_correct_gain={args.selection_min_correct_gain} "
        f"delta_multipliers={delta_multipliers} eval_mode={args.eval_mode}"
    )
    for idx, cand in enumerate(candidates):
        print(
            f"[DEBUG][GenerateValidate][Candidate {idx}] "
            f"layer={cand['layer_idx']} expert={cand['expert_idx']} direction={cand['direction']} "
            f"delta={cand['delta']:.8f} abs_delta={abs(cand['delta']):.8f} "
            f"layer_pos_score={cand['pos_auto_score']:.8f} layer_neg_score={cand['neg_auto_score']:.8f}"
        )

    current_correct, total = count_correct_answers(model, tokenizer, validation_dataset, eval_mode=args.eval_mode)
    current_acc = current_correct / max(total, 1)
    print(f"[DEBUG][GenerateValidate] baseline validation correct={current_correct}/{total} acc={current_acc:.6f}")

    selected = []
    remaining = list(candidates)
    max_selected = max(args.selection_max_selected, 0)
    min_gain = args.selection_min_correct_gain

    for step in range(max_selected):
        best = None
        best_correct = current_correct
        best_acc = current_acc

        print(f"[DEBUG][GenerateValidate][Step {step + 1}] evaluating {len(remaining)} remaining candidates")
        for cand in remaining:
            for multiplier in delta_multipliers:
                tested_delta = cand["delta"] * multiplier
                if abs(tested_delta) <= 1e-12:
                    continue

                bias_snapshot = clone_gate_bias(model, cand["layer_idx"])
                try:
                    add_gate_bias_delta(model, cand["layer_idx"], cand["expert_idx"], tested_delta)
                    cand_correct, cand_total = count_correct_answers(model, tokenizer, validation_dataset, eval_mode=args.eval_mode)
                finally:
                    restore_gate_bias(model, cand["layer_idx"], bias_snapshot)

                cand_acc = cand_correct / max(cand_total, 1)
                gain = cand_correct - current_correct
                print(
                    f"[DEBUG][GenerateValidate][Eval] layer={cand['layer_idx']} expert={cand['expert_idx']} "
                    f"direction={cand['direction']} base_delta={cand['delta']:.8f} "
                    f"multiplier={multiplier:.6f} tested_delta={tested_delta:.8f} "
                    f"correct={cand_correct}/{cand_total} acc={cand_acc:.6f} gain_correct={gain}"
                )

                if gain > (best_correct - current_correct):
                    best = dict(cand)
                    best["base_delta"] = cand["delta"]
                    best["delta_multiplier"] = multiplier
                    best["delta"] = tested_delta
                    best_correct = cand_correct
                    best_acc = cand_acc

        if best is None or (best_correct - current_correct) < min_gain:
            print(
                f"[DEBUG][GenerateValidate][Step {step + 1}] stop: "
                f"best_gain_correct={best_correct - current_correct} < min_correct_gain={min_gain}"
            )
            break

        add_gate_bias_delta(model, best["layer_idx"], best["expert_idx"], best["delta"])
        selected.append(best)
        remaining = [
            cand for cand in remaining
            if not (cand["layer_idx"] == best["layer_idx"] and cand["expert_idx"] == best["expert_idx"] and cand["direction"] == best["direction"])
        ]
        print(
            f"[DEBUG][GenerateValidate][Accept] layer={best['layer_idx']} expert={best['expert_idx']} "
            f"direction={best['direction']} base_delta={best.get('base_delta', best['delta']):.8f} "
            f"multiplier={best.get('delta_multiplier', 1.0):.6f} delta={best['delta']:.8f} "
            f"correct {current_correct}/{total} -> {best_correct}/{total} "
            f"acc {current_acc:.6f} -> {best_acc:.6f}"
        )
        current_correct = best_correct
        current_acc = best_acc

    pos_layers = sorted({cand["layer_idx"] for cand in selected if cand["direction"] == "pos"})
    neg_layers = sorted({cand["layer_idx"] for cand in selected if cand["direction"] == "neg"})
    print(
        f"[DEBUG][GenerateValidate] selected_count={len(selected)} "
        f"final_validation_correct={current_correct}/{total} acc={current_acc:.6f}"
    )
    print(f"[DEBUG][GenerateValidate] final positive layers: count={len(pos_layers)} layers={pos_layers}")
    print(f"[DEBUG][GenerateValidate] final negative layers: count={len(neg_layers)} layers={neg_layers}")
    print(
        "[DEBUG][GenerateValidate] selected candidates: "
        + str([
            (
                c["layer_idx"],
                c["expert_idx"],
                c["direction"],
                c.get("base_delta", c["delta"]),
                c.get("delta_multiplier", 1.0),
                c["delta"],
            )
            for c in selected
        ])
    )
    args.generate_validate_selected_count = len(selected)
    args.skip_after_evaluation = len(selected) == 0
    if args.skip_after_evaluation:
        print(
            "[DEBUG][GenerateValidate] no candidate reached min_correct_gain; "
            "skipping final AFTER dataset evaluation because no bias update was applied"
        )
    return selected, pos_layers, neg_layers


def compute_candidate_stability(cand):
    signal_sum = float(cand.get("importance_sum", 0.0))
    signal_sum_abs = float(cand.get("importance_sum_abs", 0.0))
    signal_sum_sq = float(cand.get("importance_sum_sq", 0.0))
    eps = 1e-12
    cancellation = abs(signal_sum) / max(signal_sum_abs, eps)
    strength = abs(signal_sum) / max(signal_sum_sq, eps) ** 0.5
    return signal_sum, signal_sum_abs, signal_sum_sq, cancellation, strength


def compute_gap_safe_global_alpha(candidates, args):
    if not candidates:
        return 0.0, {
            "selected_min_safe_gap": 0.0,
            "selected_max_abs_delta": 0.0,
            "gap_fraction": getattr(args, "direct_delta_gap_fraction", 0.4),
            "alpha_max": getattr(args, "direct_delta_alpha_max", 0.5),
            "raw_alpha": 0.0,
        }

    min_safe_gap = min(max(float(cand.get("safe_gap", 0.0)), 1e-8) for cand in candidates)
    max_abs_delta = max(abs(float(cand.get("delta", 0.0))) for cand in candidates)
    gap_fraction = float(getattr(args, "direct_delta_gap_fraction", 0.4))
    alpha_max = float(getattr(args, "direct_delta_alpha_max", 0.5))
    raw_alpha = gap_fraction * min_safe_gap / max(max_abs_delta, 1e-12)
    alpha = min(alpha_max, raw_alpha)
    return alpha, {
        "selected_min_safe_gap": min_safe_gap,
        "selected_max_abs_delta": max_abs_delta,
        "gap_fraction": gap_fraction,
        "alpha_max": alpha_max,
        "raw_alpha": raw_alpha,
    }


def compute_direct_applied_delta(cand, args):
    policy = getattr(args, "direct_delta_policy", "multiplier") if args is not None else "multiplier"

    if policy == "multiplier":
        return cand["delta"] * getattr(args, "direct_delta_multiplier", 1.0), {
            "policy": policy,
            "delta_multiplier": getattr(args, "direct_delta_multiplier", 1.0),
        }

    if policy == "fisher_margin":
        signal_sum = float(cand.get("importance_sum", 0.0))
        signal_sum_sq = float(cand.get("importance_sum_sq", 0.0))
        damping = float(getattr(args, "direct_delta_l2", 1e-3))
        eta = float(getattr(args, "direct_delta_eta", 1.0))
        gap_cap = float(getattr(args, "direct_delta_gap_cap", 0.5))
        safe_gap = max(float(cand.get("safe_gap", 0.0)), 1e-8)
        cap = max(gap_cap * safe_gap, 0.0)

        # The margin objective should be increased. Existing margin-contrast scores
        # use the historical loss-minimization sign, so this policy takes the
        # ascent step by negating the accumulated candidate signal.
        raw_step = -eta * signal_sum / max(signal_sum_sq + damping, 1e-12)
        applied_delta = max(-cap, min(cap, raw_step)) if cap > 0 else raw_step
        return applied_delta, {
            "policy": policy,
            "importance_sum": signal_sum,
            "importance_sum_sq": signal_sum_sq,
            "damping": damping,
            "eta": eta,
            "gap_cap": gap_cap,
            "safe_gap": safe_gap,
            "raw_step": raw_step,
            "cap": cap,
        }

    if policy == "stability_damped":
        signal_sum, signal_sum_abs, signal_sum_sq, cancellation, strength = compute_candidate_stability(cand)
        strength_denom = max(float(cand.get("_selected_max_strength", strength)), 1e-12)
        strength_norm_raw = max(0.0, min(1.0, strength / strength_denom))
        strength_temp = float(getattr(args, "direct_delta_strength_temp", 1.0))
        strength_norm = strength_norm_raw ** (1.0 / strength_temp)
        stability_raw = cancellation * strength_norm
        min_stability = float(getattr(args, "direct_delta_min_stability", 0.0))
        max_damping = float(getattr(args, "direct_delta_max_damping", 0.5))
        stability = stability_raw if stability_raw >= min_stability else 0.0
        damping = max_damping * stability
        applied_delta = -cand["delta"] * damping
        return applied_delta, {
            "policy": policy,
            "importance_sum": signal_sum,
            "importance_sum_abs": signal_sum_abs,
            "importance_sum_sq": signal_sum_sq,
            "cancellation": cancellation,
            "strength": strength,
            "strength_norm_raw": strength_norm_raw,
            "strength_temp": strength_temp,
            "strength_norm": strength_norm,
            "stability_raw": stability_raw,
            "min_stability": min_stability,
            "max_damping": max_damping,
            "damping": damping,
        }

    if policy == "gap_safe_global":
        alpha = float(cand.get("_gap_safe_global_alpha", 0.0))
        applied_delta = -cand["delta"] * alpha
        return applied_delta, {
            "policy": policy,
            "alpha": alpha,
            "selected_min_safe_gap": float(cand.get("_selected_min_safe_gap", 0.0)),
            "selected_max_abs_delta": float(cand.get("_selected_max_abs_delta", 0.0)),
            "gap_fraction": float(cand.get("_gap_fraction", 0.0)),
            "alpha_max": float(cand.get("_alpha_max", 0.0)),
            "raw_alpha": float(cand.get("_raw_alpha", 0.0)),
        }

    raise ValueError(f"Unsupported direct_delta_policy: {policy}")


def apply_direct_candidate_limit(model, candidates, max_candidates, args=None):
    candidates = sorted(candidates, key=lambda item: abs(item["delta"]), reverse=True)
    if max_candidates > 0:
        candidates = candidates[:max_candidates]

    selected_strengths = [compute_candidate_stability(cand)[4] for cand in candidates]
    selected_max_strength = max(selected_strengths) if selected_strengths else 0.0
    gap_safe_alpha, gap_safe_debug = compute_gap_safe_global_alpha(candidates, args)
    for cand in candidates:
        cand["_selected_max_strength"] = selected_max_strength
        cand["_gap_safe_global_alpha"] = gap_safe_alpha
        cand["_selected_min_safe_gap"] = gap_safe_debug["selected_min_safe_gap"]
        cand["_selected_max_abs_delta"] = gap_safe_debug["selected_max_abs_delta"]
        cand["_gap_fraction"] = gap_safe_debug["gap_fraction"]
        cand["_alpha_max"] = gap_safe_debug["alpha_max"]
        cand["_raw_alpha"] = gap_safe_debug["raw_alpha"]

    policy = getattr(args, "direct_delta_policy", "multiplier") if args is not None else "multiplier"
    delta_multiplier = getattr(args, "direct_delta_multiplier", 1.0) if args is not None else 1.0
    print(
        f"[DEBUG][DirectTopN] applying candidate_count={len(candidates)} "
        f"max_candidates={max_candidates} direct_delta_policy={policy} "
        f"delta_multiplier={delta_multiplier:.6f}"
    )
    for idx, cand in enumerate(candidates):
        applied_delta, delta_debug = compute_direct_applied_delta(cand, args)
        delta_debug_str = " ".join(
            f"{key}={value:.8f}" if isinstance(value, float) else f"{key}={value}"
            for key, value in delta_debug.items()
        )
        print(
            f"[DEBUG][DirectTopN][Candidate {idx}] "
            f"layer={cand['layer_idx']} expert={cand['expert_idx']} direction={cand['direction']} "
            f"delta={cand['delta']:.8f} abs_delta={abs(cand['delta']):.8f} "
            f"applied_delta={applied_delta:.8f} "
            f"layer_pos_score={cand['pos_auto_score']:.8f} layer_neg_score={cand['neg_auto_score']:.8f} "
            f"{delta_debug_str}"
        )
        cand["applied_delta"] = applied_delta
        add_gate_bias_delta(model, cand["layer_idx"], cand["expert_idx"], applied_delta)

    pos_layers = sorted({cand["layer_idx"] for cand in candidates if cand["direction"] == "pos"})
    neg_layers = sorted({cand["layer_idx"] for cand in candidates if cand["direction"] == "neg"})
    print(f"[DEBUG][DirectTopN] final positive layers: count={len(pos_layers)} layers={pos_layers}")
    print(f"[DEBUG][DirectTopN] final negative layers: count={len(neg_layers)} layers={neg_layers}")
    return candidates, pos_layers, neg_layers


def compute_expert_importance(
    model,
    tokenizer,
    dataset,
    loss_top_k='top_k_x2',
    generation_max_new_tokens=20,
    loss_position='extracted_generated_answer',
    loss_type='ce',
    margin_tau=0.0,
    lambda_ce=0.5,
    boundary_scale=1.0,
    boundary_weight_min=0.1,
    boundary_weight_max=1.0,
    importance_type='legacy',
    router_reach_temp=1.0,
    router_reach_bias=0.0,
):
    model.eval()
    num_layers = len(model.model.layers)

    expert_importance = [defaultdict(float) for _ in range(num_layers)]
    all_router_logits = defaultdict(list)
    importance_diagnostics = [defaultdict(lambda: {
        "sum": 0.0,
        "sum_sq": 0.0,
        "sum_abs": 0.0,
        "sum_reach": 0.0,
        "pos": 0,
        "neg": 0,
        "nonzero": 0,
    }) for _ in range(num_layers)]
    loss_diagnostics = []
    letter_token_ids = get_letter_token_ids(tokenizer)
    stop_words = ['<|im_end|>', '<|im_start|>']
    if importance_type not in {
        'legacy',
        'router_bias_contrast',
        'router_bias_contrast_loss',
        'router_bias_contrast_margin',
        'router_margin_grad',
    }:
        raise ValueError(f"Unsupported importance_type: {importance_type}")
    if router_reach_temp <= 0:
        raise ValueError("--router_reach_temp must be > 0")

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
        extracted_answer = ''
        extracted_gen_idx = None

        if loss_position == 'extracted_generated_answer':
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
        elif loss_position == 'canonical_answer_slot':
            conditioning_ids = prompt_ids
            print(f"[DEBUG] Example {i+1}: using canonical answer slot")
        else:
            raise ValueError(f"Unsupported loss_position: {loss_position}")

        pred_pos = conditioning_ids.shape[1] - 1

        hidden_states_with_grad = {}
        mlp_inputs = {}
        router_logits_with_grad = {}

        def make_backward_hook(layer_idx):
            def hook(module, grad_input, grad_output):
                if grad_output[0] is not None:
                    hidden_states_with_grad[layer_idx] = grad_output[0].clone()
            return hook

        def make_forward_hook(layer_idx):
            def hook(module, input, output):
                mlp_inputs[layer_idx] = input[0].detach().clone()
            return hook

        def make_router_forward_hook(layer_idx):
            def hook(module, input, output):
                output.retain_grad()
                router_logits_with_grad[layer_idx] = output
            return hook

        hooks = []
        for layer_idx in range(num_layers):
            if importance_type in {
                'legacy',
                'router_bias_contrast',
                'router_bias_contrast_loss',
                'router_bias_contrast_margin',
            }:
                backward_hook = model.model.layers[layer_idx].mlp.register_full_backward_hook(make_backward_hook(layer_idx))
                forward_hook = model.model.layers[layer_idx].mlp.register_forward_hook(make_forward_hook(layer_idx))
                hooks.append(backward_hook)
                hooks.append(forward_hook)
            else:
                router_hook = model.model.layers[layer_idx].mlp.gate.register_forward_hook(make_router_forward_hook(layer_idx))
                hooks.append(router_hook)

        outputs = model(input_ids=conditioning_ids, output_hidden_states=True)

        # Build A/B/C/D logits for the selected answer prediction position.
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

        loss, loss_stats = compute_answer_loss(
            abcd_logits,
            correct_idx,
            loss_type,
            tau=margin_tau,
            lambda_ce=lambda_ce,
            boundary_scale=boundary_scale,
            boundary_weight_min=boundary_weight_min,
            boundary_weight_max=boundary_weight_max,
        )
        wrong_mask = torch.ones(abcd_logits.shape[0], dtype=torch.bool, device=abcd_logits.device)
        wrong_mask[correct_idx] = False
        smooth_margin = abcd_logits[correct_idx].float() - torch.logsumexp(abcd_logits[wrong_mask].float(), dim=0)
        router_margin_objective = loss_stats['boundary_weight'].to(smooth_margin.device) * smooth_margin
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
        margin_value = loss_stats['margin'].item()
        if margin_value > boundary_scale:
            margin_bucket = "easy_correct"
        elif margin_value >= 0.0:
            margin_bucket = "close_correct"
        elif margin_value >= -boundary_scale:
            margin_bucket = "flippable_wrong"
        else:
            margin_bucket = "hard_wrong"
        loss_diagnostics.append({
            "pred_correct": pred_letter == correct_answer,
            "p_correct": p_correct.item(),
            "margin": margin_value,
            "ce_loss": loss_stats['ce_loss'].item(),
            "soft_margin_loss": loss_stats['soft_margin_loss'].item(),
            "boundary_weighted_margin_loss": loss_stats['boundary_weighted_margin_loss'].item(),
            "boundary_weight": loss_stats['boundary_weight'].item(),
            "loss": loss.item(),
            "margin_bucket": margin_bucket,
        })
        if loss_position == 'extracted_generated_answer':
            print(f"[DEBUG] Predicted={pred_letter}, extracted_answer={extracted_answer}, gold={correct_answer}")
        else:
            print(f"[DEBUG] Predicted={pred_letter}, gold={correct_answer}")
        print(
            f"[DEBUG] Loss ({loss_type}, position={loss_position}, importance={importance_type}): "
            f"p_correct={p_correct.item():.6f}, "
            f"margin={loss_stats['margin'].item():.6f}, "
            f"smooth_margin={smooth_margin.detach().item():.6f}, "
            f"margin_bucket={margin_bucket}, "
            f"ce={loss_stats['ce_loss'].item():.6f}, "
            f"soft_margin={loss_stats['soft_margin_loss'].item():.6f}, "
            f"boundary_weight={loss_stats['boundary_weight'].item():.6f}, "
            f"boundary_weighted_margin={loss_stats['boundary_weighted_margin_loss'].item():.6f}, "
            f"loss={loss.item():.6f}"
        )

        model.zero_grad()
        if importance_type in {'legacy', 'router_bias_contrast_loss'}:
            loss.backward()
        else:
            router_margin_objective.backward()
        for hook in hooks:
            hook.remove()

        for layer_idx in range(num_layers):
            layer = model.model.layers[layer_idx]
            grad_token_idx = pred_pos

            # Determine effective top_k based on loss_top_k parameter
            base_top_k = layer.mlp.top_k
            num_experts = layer.mlp.num_experts
            if loss_top_k == 'top_k':
                effective_top_k = base_top_k
            elif loss_top_k == 'top_k_x2':
                effective_top_k = base_top_k * 2
            else:  # 'all'
                effective_top_k = num_experts

            if importance_type == 'router_margin_grad':
                if layer_idx not in router_logits_with_grad:
                    print(f"[DEBUG] Layer {layer_idx}: no router logits captured, skipping")
                    continue

                router_logits_raw = router_logits_with_grad[layer_idx]
                if router_logits_raw.dim() == 3:
                    router_logits = router_logits_raw.reshape(-1, router_logits_raw.shape[-1])
                    router_grads = router_logits_raw.grad.reshape(-1, router_logits_raw.shape[-1])
                else:
                    router_logits = router_logits_raw
                    router_grads = router_logits_raw.grad

                if router_grads is None:
                    print(f"[DEBUG] Layer {layer_idx}: no router gradients, skipping")
                    continue

                logits_token = router_logits[grad_token_idx].detach().float()
                grads_token = router_grads[grad_token_idx].detach().float()
                sorted_logits, _ = torch.sort(logits_token, descending=True)
                top_k = int(getattr(layer.mlp, "top_k", getattr(model.config, "num_experts_per_tok", 4)))
                kth_index = min(max(top_k - 1, 0), sorted_logits.numel() - 1)
                kth_logit = sorted_logits[kth_index]
                reach = torch.sigmoid((logits_token + router_reach_bias - kth_logit) / router_reach_temp)
                scores = grads_token * reach

                all_router_logits[layer_idx].append(logits_token.detach().cpu())
                for expert_idx in range(num_experts):
                    score = scores[expert_idx].item()
                    expert_importance[layer_idx][expert_idx] += score
                    if abs(score) > 1e-12:
                        diag = importance_diagnostics[layer_idx][expert_idx]
                        diag["sum"] += score
                        diag["sum_sq"] += score * score
                        diag["sum_abs"] += abs(score)
                        diag["sum_reach"] += reach[expert_idx].item()
                        diag["nonzero"] += 1
                        if score > 0:
                            diag["pos"] += 1
                        elif score < 0:
                            diag["neg"] += 1
                continue

            if layer_idx not in hidden_states_with_grad:
                print(f"[DEBUG] Layer {layer_idx}: no gradients, skipping")
                continue

            g = hidden_states_with_grad[layer_idx]
            hidden_flat = mlp_inputs[layer_idx]

            _, _, hidden_dim = hidden_flat.shape
            hidden_flat = hidden_flat.view(-1, hidden_dim)
            g_flat = g.view(-1, hidden_dim).to(hidden_flat.device)

            router_logits = layer.mlp.gate(hidden_flat)
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)

            weights, selected = torch.topk(routing_weights, effective_top_k, dim=-1)

            if layer.mlp.norm_topk_prob:
                weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-6)

            weights = weights.to(hidden_flat.device)
            selected = selected.to(hidden_flat.device)

            # Use exactly the same position as the selected answer predictor.
            g_token = g_flat[grad_token_idx]
            selected_experts = selected[grad_token_idx]
            selected_weights = weights[grad_token_idx].float()
            expert_outputs = []

            with torch.no_grad():
                for k_idx in range(effective_top_k):
                    expert_idx = selected_experts[k_idx].item()
                    h_i = layer.mlp.experts[expert_idx](
                        hidden_flat[grad_token_idx : grad_token_idx + 1]
                    ).to(hidden_flat.device)
                    expert_outputs.append(h_i.squeeze(0))

            expert_outputs = torch.stack(expert_outputs, dim=0)
            selected_weights = selected_weights.to(expert_outputs.device)
            y_token = torch.sum(selected_weights.unsqueeze(-1) * expert_outputs, dim=0)

            for k_idx in range(effective_top_k):
                expert_idx = selected_experts[k_idx].item()
                w_i = selected_weights[k_idx]
                h_i = expert_outputs[k_idx]

                if importance_type == 'legacy':
                    importance_vector = w_i * h_i
                elif importance_type in {'router_bias_contrast', 'router_bias_contrast_loss', 'router_bias_contrast_margin'}:
                    importance_vector = w_i * (h_i - y_token)
                else:
                    raise ValueError(f"Unsupported importance_type in hidden-output branch: {importance_type}")

                importance = -torch.dot(
                    g_token.float(),
                    importance_vector.to(g_token.device).float(),
                ).item()
                expert_importance[layer_idx][expert_idx] += importance
                if abs(importance) > 1e-12:
                    diag = importance_diagnostics[layer_idx][expert_idx]
                    diag["sum"] += importance
                    diag["sum_sq"] += importance * importance
                    diag["sum_abs"] += abs(importance)
                    diag["nonzero"] += 1
                    if importance > 0:
                        diag["pos"] += 1
                    elif importance < 0:
                        diag["neg"] += 1

            all_router_logits[layer_idx].append(router_logits[grad_token_idx].detach().cpu())

        model.zero_grad()

    if loss_diagnostics:
        total = len(loss_diagnostics)
        correct_count = sum(1 for item in loss_diagnostics if item["pred_correct"])
        print(
            f"[DEBUG][LossSummary] samples_used={total} "
            f"canonical_or_extracted_top1_correct={correct_count}/{total} "
            f"accuracy={correct_count / total:.6f}"
        )
        for bucket in ["easy_correct", "close_correct", "flippable_wrong", "hard_wrong"]:
            bucket_items = [item for item in loss_diagnostics if item["margin_bucket"] == bucket]
            if not bucket_items:
                print(f"[DEBUG][LossSummary][{bucket}] count=0")
                continue

            def mean_value(key):
                return sum(item[key] for item in bucket_items) / len(bucket_items)

            print(
                f"[DEBUG][LossSummary][{bucket}] count={len(bucket_items)} "
                f"mean_margin={mean_value('margin'):.6f} "
                f"mean_p_correct={mean_value('p_correct'):.6f} "
                f"mean_soft_margin={mean_value('soft_margin_loss'):.6f} "
                f"mean_boundary_weight={mean_value('boundary_weight'):.6f} "
                f"mean_weighted_loss={mean_value('boundary_weighted_margin_loss'):.6f} "
                f"mean_actual_loss={mean_value('loss'):.6f}"
            )

    if importance_type == 'router_margin_grad':
        print("[DEBUG][RouterMarginGradSummary] objective=boundary_weight * (gold_logit - logsumexp(wrong_logits))")
        print(
            f"[DEBUG][RouterMarginGradSummary] reach=sigmoid((router_logit + {router_reach_bias:.6f} - kth_router_logit) / {router_reach_temp:.6f})"
        )
        for layer_idx in range(num_layers):
            if not expert_importance[layer_idx]:
                continue
            layer_scores = torch.zeros(model.model.layers[layer_idx].mlp.num_experts)
            for expert_idx, score in expert_importance[layer_idx].items():
                layer_scores[expert_idx] = score
            sorted_indices = torch.argsort(layer_scores, descending=True)
            top_3 = []
            bottom_3 = []
            for idx in sorted_indices[:3]:
                diag = importance_diagnostics[layer_idx].get(idx.item(), None)
                nonzero = diag["nonzero"] if diag else 0
                consistency = (diag["pos"] / nonzero) if diag and nonzero else 0.0
                mean_reach = (diag["sum_reach"] / nonzero) if diag and nonzero else 0.0
                top_3.append((idx.item(), layer_scores[idx].item(), consistency, mean_reach, nonzero))
            for idx in sorted_indices[-3:]:
                diag = importance_diagnostics[layer_idx].get(idx.item(), None)
                nonzero = diag["nonzero"] if diag else 0
                consistency = (diag["neg"] / nonzero) if diag and nonzero else 0.0
                mean_reach = (diag["sum_reach"] / nonzero) if diag and nonzero else 0.0
                bottom_3.append((idx.item(), layer_scores[idx].item(), consistency, mean_reach, nonzero))
            print(
                f"[DEBUG][RouterMarginGradSummary][Layer {layer_idx}] "
                f"score_min={layer_scores.min().item():.8f} "
                f"score_max={layer_scores.max().item():.8f} "
                f"score_mean={layer_scores.mean().item():.8f} "
                f"top3=(expert,score,pos_consistency,mean_reach,nonzero)={top_3} "
                f"bottom3=(expert,score,neg_consistency,mean_reach,nonzero)={bottom_3}"
            )

    return expert_importance, all_router_logits, importance_diagnostics

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


def iter_dataset_examples(dataset, max_samples=None):
    total = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    for i in range(total):
        yield dataset[i]


def compute_answer_margins(model, tokenizer, dataset, letter_token_ids, max_samples=None):
    margins = []
    correct_flags = []
    total = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    for example in iter_dataset_examples(dataset, max_samples):
        prompt = format_mmlu_question(example)
        correct_idx = int(example["answer"])
        chat_text = build_chat_texts(tokenizer, [prompt])[0]
        inputs = tokenizer(
            [chat_text],
            return_tensors="pt",
            add_special_tokens=False,
        ).to(model.device)

        with torch.no_grad():
            outputs = model(input_ids=inputs["input_ids"])
            logits_pos = outputs.logits[0, inputs["input_ids"].shape[1] - 1, :]
            abcd_logits = build_abcd_logits(logits_pos, letter_token_ids).float()

        wrong_mask = torch.ones(abcd_logits.shape[0], dtype=torch.bool, device=abcd_logits.device)
        wrong_mask[correct_idx] = False
        margin = (abcd_logits[correct_idx] - torch.max(abcd_logits[wrong_mask])).item()
        pred_idx = int(torch.argmax(abcd_logits).item())
        margins.append(margin)
        correct_flags.append(pred_idx == correct_idx)

    if len(margins) != total:
        print(f"[DEBUG][CausalFilter] warning: expected {total} margin samples, got {len(margins)}")
    return margins, correct_flags


def mean_or_zero(values):
    return sum(values) / len(values) if values else 0.0


def causal_filter_v6_candidates(
    model,
    tokenizer,
    dataset,
    layer,
    layer_idx,
    selected_pos_final,
    selected_neg_final,
    delta_bias,
    args,
):
    if not args.causal_filter_delta_margin:
        return selected_pos_final, selected_neg_final

    candidate_indices = torch.cat([selected_pos_final, selected_neg_final]).unique()
    if candidate_indices.numel() == 0:
        return selected_pos_final, selected_neg_final

    letter_token_ids = get_letter_token_ids(tokenizer)
    max_samples = None if args.causal_filter_samples <= 0 else args.causal_filter_samples
    kept_indices = []

    print(
        f"[DEBUG][CausalFilter][Layer {layer_idx}] start: "
        f"candidates={candidate_indices.tolist()} "
        f"samples={'all' if max_samples is None else max_samples} "
        f"min_delta={args.causal_filter_min_delta:.8f}"
    )

    for expert_idx_tensor in candidate_indices:
        expert_idx = int(expert_idx_tensor.item())
        signed_delta = delta_bias[expert_idx].to(dtype=layer.mlp.gate.weight.dtype, device=layer.mlp.gate.weight.device)

        if layer.mlp.gate.bias is None:
            layer.mlp.gate.bias = nn.Parameter(torch.zeros(
                layer.mlp.num_experts,
                dtype=layer.mlp.gate.weight.dtype,
                device=layer.mlp.gate.weight.device,
            ))

        base_margins, base_correct = compute_answer_margins(
            model, tokenizer, dataset, letter_token_ids, max_samples=max_samples
        )

        with torch.no_grad():
            layer.mlp.gate.bias[expert_idx] += signed_delta

        try:
            biased_margins, biased_correct = compute_answer_margins(
                model, tokenizer, dataset, letter_token_ids, max_samples=max_samples
            )
        finally:
            with torch.no_grad():
                layer.mlp.gate.bias[expert_idx] -= signed_delta

        delta_margins = [biased - base for base, biased in zip(base_margins, biased_margins)]
        mean_delta = mean_or_zero(delta_margins)
        improved_count = sum(1 for delta in delta_margins if delta > 0.0)
        worsened_count = sum(1 for delta in delta_margins if delta < 0.0)
        unchanged_count = len(delta_margins) - improved_count - worsened_count
        base_acc = sum(base_correct) / len(base_correct) if base_correct else 0.0
        biased_acc = sum(biased_correct) / len(biased_correct) if biased_correct else 0.0
        keep = mean_delta >= args.causal_filter_min_delta

        print(
            f"[DEBUG][CausalFilter][Layer {layer_idx}][Expert {expert_idx}] "
            f"signed_delta={float(signed_delta.item()):.8f} "
            f"base_mean_margin={mean_or_zero(base_margins):.8f} "
            f"biased_mean_margin={mean_or_zero(biased_margins):.8f} "
            f"mean_delta_margin={mean_delta:.8f} "
            f"base_acc={base_acc:.6f} biased_acc={biased_acc:.6f} "
            f"improved={improved_count} worsened={worsened_count} unchanged={unchanged_count} "
            f"keep={keep}"
        )

        if keep:
            kept_indices.append(expert_idx)

    kept = torch.tensor(kept_indices, dtype=torch.long, device=selected_pos_final.device)
    filtered_pos = selected_pos_final[torch.isin(selected_pos_final, kept)] if selected_pos_final.numel() > 0 else selected_pos_final
    filtered_neg = selected_neg_final[torch.isin(selected_neg_final, kept)] if selected_neg_final.numel() > 0 else selected_neg_final

    print(
        f"[DEBUG][CausalFilter][Layer {layer_idx}] result: "
        f"pos_before={selected_pos_final.tolist()} pos_after={filtered_pos.tolist()} "
        f"neg_before={selected_neg_final.tolist()} neg_after={filtered_neg.tolist()}"
    )
    return filtered_pos, filtered_neg


def apply_expert_bias_v6(model, expert_importance, all_router_logits, importance_diagnostics=None, tokenizer=None, calibration_dataset=None, args=None):
    print("[DEBUG] Applying bias by formula v6 (gap-normalized sparse bias)")
    num_layers = len(model.model.layers)
    layer_stats = []
    positive_bias_layers_applied = []
    negative_bias_layers_applied = []
    validation_candidates = []
    direct_candidates = []
    use_generate_validate = args is not None and args.selection_type == 'generate_validate'
    use_direct_candidate_limit = (
        args is not None
        and args.selection_type == 'direct'
        and getattr(args, "direct_max_candidates", 0) > 0
    )
    if use_generate_validate and (tokenizer is None or calibration_dataset is None):
        raise ValueError("selection_type=generate_validate requires tokenizer and calibration_dataset")

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
        safe_gap = stats["safe_gap"]
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

        selected_indices = torch.cat([selected_pos_final, selected_neg_final]).unique()
        sparse_importances = torch.zeros_like(importances)
        sparse_importances[selected_indices] = importances[selected_indices]
        delta_bias = eps * sparse_importances

        if args is not None and args.causal_filter_delta_margin:
            if tokenizer is None or calibration_dataset is None:
                raise ValueError("causal_filter_delta_margin requires tokenizer and calibration_dataset")
            selected_pos_final, selected_neg_final = causal_filter_v6_candidates(
                model,
                tokenizer,
                calibration_dataset,
                layer,
                layer_idx,
                selected_pos_final,
                selected_neg_final,
                delta_bias,
                args,
            )
            selected_indices = torch.cat([selected_pos_final, selected_neg_final]).unique()
            sparse_importances = torch.zeros_like(importances)
            sparse_importances[selected_indices] = importances[selected_indices]
            delta_bias = eps * sparse_importances

        pos_updates = [(idx.item(), delta_bias[idx].item()) for idx in selected_pos_final]
        neg_updates = [(idx.item(), delta_bias[idx].item()) for idx in selected_neg_final]

        def make_candidate(idx, direction):
            expert_idx = idx.item()
            diag = (
                importance_diagnostics[layer_idx].get(expert_idx, {})
                if importance_diagnostics is not None
                else {}
            )
            return {
                "layer_idx": layer_idx,
                "expert_idx": expert_idx,
                "direction": direction,
                "delta": delta_bias[idx].item(),
                "normalized_importance": importances[idx].item(),
                "importance_sum": float(diag.get("sum", expert_importance[layer_idx].get(expert_idx, 0.0))),
                "importance_sum_sq": float(diag.get("sum_sq", 0.0)),
                "importance_sum_abs": float(diag.get("sum_abs", 0.0)),
                "importance_nonzero": int(diag.get("nonzero", 0)),
                "g1": g1,
                "gk": gk,
                "safe_gap": safe_gap,
                "eps": eps,
                "pos_auto_score": pos_auto_score,
                "neg_auto_score": neg_auto_score,
            }

        if use_generate_validate:
            if not args.selection_negative_only:
                for idx in selected_pos_final:
                    validation_candidates.append(make_candidate(idx, "pos"))
            for idx in selected_neg_final:
                validation_candidates.append(make_candidate(idx, "neg"))
        elif use_direct_candidate_limit:
            for idx in selected_pos_final:
                direct_candidates.append(make_candidate(idx, "pos"))
            for idx in selected_neg_final:
                direct_candidates.append(make_candidate(idx, "neg"))
        else:
            if selected_pos_final.numel() > 0:
                positive_bias_layers_applied.append(layer_idx)
            if selected_neg_final.numel() > 0:
                negative_bias_layers_applied.append(layer_idx)

            with torch.no_grad():
                if layer.mlp.gate.bias is None:
                    layer.mlp.gate.bias = nn.Parameter(torch.zeros(num_experts, dtype=layer.mlp.gate.weight.dtype, device=layer.mlp.gate.weight.device))
                layer.mlp.gate.bias += delta_bias.to(dtype=layer.mlp.gate.bias.dtype, device=layer.mlp.gate.bias.device)

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
        if use_generate_validate:
            print(f"[DEBUG][Layer {layer_idx}] candidate experts for generate-validate: pos={pos_updates if not args.selection_negative_only else []} neg={neg_updates}")
        elif use_direct_candidate_limit:
            print(f"[DEBUG][Layer {layer_idx}] candidate experts for direct top-N: pos={pos_updates} neg={neg_updates}")
        else:
            print(f"[DEBUG][Layer {layer_idx}] sparse experts for bias update: pos={pos_updates} neg={neg_updates}")

    if use_generate_validate:
        _, positive_bias_layers_applied, negative_bias_layers_applied = generate_validate_select_and_apply(
            model,
            tokenizer,
            calibration_dataset,
            validation_candidates,
            args,
        )
    elif use_direct_candidate_limit:
        _, positive_bias_layers_applied, negative_bias_layers_applied = apply_direct_candidate_limit(
            model,
            direct_candidates,
            args.direct_max_candidates,
            args,
        )

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


def run_calibration_method(model, tokenizer, calibration_dataset, expert_importance, all_router_logits, importance_diagnostics, args):
    if args.method == 'baseline':
        return model
    if args.method == 'original':
        return apply_expert_bias(model, expert_importance, all_router_logits)
    if args.method == 'v4':
        return apply_expert_bias_v4(model, expert_importance, all_router_logits)
    if args.method == 'v5':
        return apply_expert_bias_v5(model, expert_importance, all_router_logits)
    if args.method == 'v6':
        return apply_expert_bias_v6(
            model,
            expert_importance,
            all_router_logits,
            importance_diagnostics=importance_diagnostics,
            tokenizer=tokenizer,
            calibration_dataset=calibration_dataset,
            args=args,
        )
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


def evaluate_named_datasets_with_mode(model, tokenizer, named_datasets, prefix, eval_mode):
    for dataset_name, dataset in named_datasets:
        accuracy = test_accuracy(model, tokenizer, dataset, eval_mode=eval_mode)
        print(f"{prefix} {dataset_name} ({eval_mode}):", accuracy)


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
    parser.add_argument('--eval_mode', type=str, default='generation',
                       choices=['generation', 'next_token_pred'],
                       help='Evaluation/scoring mode: generation uses text generation + extraction; next_token_pred uses forced-choice A/B/C/D logits at the next token')
    parser.add_argument('--loss_top_k', type=str, default='top_k',
                       choices=['top_k', 'top_k_x2', 'all'],
                       help='Loss calculation mode: top_k (default), top_k_x2 (top_k*2), or all (num_experts)')
    parser.add_argument('--loss_position', type=str, default='extracted_generated_answer',
                       choices=['extracted_generated_answer', 'canonical_answer_slot'],
                       help='Where to compute calibration loss: old extracted generated answer position or canonical prompt answer slot')
    parser.add_argument('--loss_type', type=str, default='ce',
                       choices=['ce', 'soft_margin', 'ce_margin_hybrid', 'boundary_weighted_margin'],
                       help='Answer loss formula over A/B/C/D logits')
    parser.add_argument('--margin_tau', type=float, default=0.0,
                       help='Margin offset tau for soft_margin, ce_margin_hybrid, and boundary_weighted_margin losses')
    parser.add_argument('--lambda_ce', type=float, default=0.5,
                       help='CE weight for ce_margin_hybrid loss')
    parser.add_argument('--boundary_scale', type=float, default=1.0,
                       help='Scale s for boundary_weighted_margin: weight = exp(-abs(margin) / s)')
    parser.add_argument('--boundary_weight_min', type=float, default=0.1,
                       help='Minimum clipped example weight for boundary_weighted_margin')
    parser.add_argument('--boundary_weight_max', type=float, default=1.0,
                       help='Maximum clipped example weight for boundary_weighted_margin')
    parser.add_argument('--importance_type', type=str, default='legacy',
                       choices=[
                           'legacy',
                           'router_bias_contrast',
                           'router_bias_contrast_loss',
                           'router_bias_contrast_margin',
                           'router_margin_grad',
                       ],
                       help='Expert-importance formula: legacy contribution dot-product; router_bias_contrast_margin uses margin objective with -dot(dmargin/dy, w_e*(h_e-y)); router_bias_contrast_loss uses answer loss with -dot(dL/dy, w_e*(h_e-y)); router_bias_contrast is kept as a margin alias; router_margin_grad uses router-logit gradients')
    parser.add_argument('--router_reach_temp', type=float, default=1.0,
                       help='Temperature for router_margin_grad reach = sigmoid((router_logit + bias - kth_logit) / temp)')
    parser.add_argument('--router_reach_bias', type=float, default=0.0,
                       help='Probe bias added inside router_margin_grad reach calculation')
    parser.add_argument('--selection_type', type=str, default='direct',
                       choices=['direct', 'generate_validate'],
                       help='Final v6 candidate selection: direct applies formula-selected candidates; generate_validate greedily accepts candidates by generated-answer accuracy')
    parser.add_argument('--selection_validation_size', type=int, default=30,
                       help='Number of calibration samples from the end of the calibration set used for generate_validate selection; <=0 means all')
    parser.add_argument('--selection_max_candidates', type=int, default=12,
                       help='Maximum candidate bias updates evaluated by generate_validate, sorted by absolute delta; <=0 means all')
    parser.add_argument('--selection_max_selected', type=int, default=3,
                       help='Maximum number of candidates accepted by generate_validate')
    parser.add_argument('--selection_min_correct_gain', type=int, default=1,
                       help='Minimum validation correct-count gain required to accept a generate_validate candidate')
    parser.add_argument('--selection_delta_multipliers', type=str, default='1.0',
                       help='Comma-separated delta multipliers tested per generate_validate candidate, e.g. "-2,-1,-0.5,0.5,1,2"')
    parser.add_argument('--selection_negative_only', action='store_true',
                       help='For generate_validate, evaluate only negative/suppression candidates')
    parser.add_argument('--direct_max_candidates', type=int, default=0,
                       help='For v6 selection_type=direct, apply only the top-N sparse bias candidates by abs(delta); <=0 keeps the old behavior and applies all selected sparse candidates')
    parser.add_argument('--direct_delta_multiplier', type=float, default=1.0,
                       help='For v6 selection_type=direct with direct_max_candidates > 0, multiply each selected candidate delta by this value before applying it')
    parser.add_argument('--direct_delta_policy', type=str, default='multiplier',
                       choices=['multiplier', 'fisher_margin', 'stability_damped', 'gap_safe_global'],
                       help='For v6 direct top-N, choose how to convert selected candidate scores into applied router-bias deltas. multiplier keeps the old behavior; fisher_margin uses a sign-corrected damped margin step; stability_damped keeps base delta but flips sign and damps it by signal stability; gap_safe_global flips sign and applies one router-gap-safe alpha to all selected base deltas')
    parser.add_argument('--direct_delta_eta', type=float, default=1.0,
                       help='Step size eta for --direct_delta_policy fisher_margin')
    parser.add_argument('--direct_delta_l2', type=float, default=1e-3,
                       help='Damping lambda for --direct_delta_policy fisher_margin: delta = -eta * sum / (sum_sq + lambda), then clipped')
    parser.add_argument('--direct_delta_gap_cap', type=float, default=0.5,
                       help='Router-gap clipping fraction for --direct_delta_policy fisher_margin; applied delta is clipped to +/- gap_cap * mean top-k boundary gap')
    parser.add_argument('--direct_delta_min_stability', type=float, default=0.0,
                       help='For --direct_delta_policy stability_damped, zero out a candidate if cancellation * normalized_strength is below this threshold')
    parser.add_argument('--direct_delta_strength_temp', type=float, default=1.0,
                       help='For --direct_delta_policy stability_damped, temperature shaping for normalized strength; values >1 make weaker candidates less suppressed')
    parser.add_argument('--direct_delta_max_damping', type=float, default=0.5,
                       help='For --direct_delta_policy stability_damped, maximum multiplier applied to sign-corrected base delta')
    parser.add_argument('--direct_delta_gap_fraction', type=float, default=0.4,
                       help='For --direct_delta_policy gap_safe_global, fraction of the minimum selected router top-k boundary gap allowed for the largest applied delta')
    parser.add_argument('--direct_delta_alpha_max', type=float, default=0.5,
                       help='For --direct_delta_policy gap_safe_global, maximum global alpha in applied_delta = -alpha * base_delta')
    parser.add_argument('--causal_filter_delta_margin', action='store_true',
                       help='For v6, keep selected expert bias candidates only if temporary signed bias improves mean answer margin')
    parser.add_argument('--causal_filter_samples', type=int, default=32,
                       help='Number of calibration samples for delta-margin causal filter; use <=0 for all samples')
    parser.add_argument('--causal_filter_min_delta', type=float, default=0.0,
                       help='Minimum mean delta margin required to keep a v6 expert candidate')
    parser.add_argument('--eps_value', type=float, default=None,
                       help='Fixed epsilon value for default method (e.g., 0.005, 0.01, 0.05, 0.1, 0.2, 0.3)')
    args = parser.parse_args()

    if args.boundary_scale <= 0:
        raise ValueError("--boundary_scale must be > 0")
    if args.boundary_weight_min < 0:
        raise ValueError("--boundary_weight_min must be >= 0")
    if args.boundary_weight_max < args.boundary_weight_min:
        raise ValueError("--boundary_weight_max must be >= --boundary_weight_min")
    if args.router_reach_temp <= 0:
        raise ValueError("--router_reach_temp must be > 0")
    if args.direct_delta_eta <= 0:
        raise ValueError("--direct_delta_eta must be > 0")
    if args.direct_delta_l2 < 0:
        raise ValueError("--direct_delta_l2 must be >= 0")
    if args.direct_delta_gap_cap < 0:
        raise ValueError("--direct_delta_gap_cap must be >= 0")
    if args.direct_delta_min_stability < 0:
        raise ValueError("--direct_delta_min_stability must be >= 0")
    if args.direct_delta_strength_temp <= 0:
        raise ValueError("--direct_delta_strength_temp must be > 0")
    if args.direct_delta_max_damping < 0:
        raise ValueError("--direct_delta_max_damping must be >= 0")
    if args.direct_delta_gap_fraction < 0:
        raise ValueError("--direct_delta_gap_fraction must be >= 0")
    if args.direct_delta_alpha_max < 0:
        raise ValueError("--direct_delta_alpha_max must be >= 0")
    if args.selection_validation_size < 0:
        raise ValueError("--selection_validation_size must be >= 0")
    if args.selection_max_selected < 0:
        raise ValueError("--selection_max_selected must be >= 0")
    args.selection_delta_multipliers = parse_float_list(args.selection_delta_multipliers)


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
                "--eval_mode", args.eval_mode,
                "--loss_top_k", args.loss_top_k,
                "--loss_position", args.loss_position,
                "--loss_type", args.loss_type,
                "--margin_tau", str(args.margin_tau),
                "--lambda_ce", str(args.lambda_ce),
                "--boundary_scale", str(args.boundary_scale),
                "--boundary_weight_min", str(args.boundary_weight_min),
                "--boundary_weight_max", str(args.boundary_weight_max),
                "--importance_type", args.importance_type,
                "--router_reach_temp", str(args.router_reach_temp),
                "--router_reach_bias", str(args.router_reach_bias),
                "--selection_type", args.selection_type,
                "--selection_validation_size", str(args.selection_validation_size),
                "--selection_max_candidates", str(args.selection_max_candidates),
                "--selection_max_selected", str(args.selection_max_selected),
                "--selection_min_correct_gain", str(args.selection_min_correct_gain),
                "--selection_delta_multipliers", ",".join(str(v) for v in args.selection_delta_multipliers),
                "--direct_max_candidates", str(args.direct_max_candidates),
                "--direct_delta_multiplier", str(args.direct_delta_multiplier),
                "--direct_delta_policy", args.direct_delta_policy,
                "--direct_delta_eta", str(args.direct_delta_eta),
                "--direct_delta_l2", str(args.direct_delta_l2),
                "--direct_delta_gap_cap", str(args.direct_delta_gap_cap),
                "--direct_delta_min_stability", str(args.direct_delta_min_stability),
                "--direct_delta_strength_temp", str(args.direct_delta_strength_temp),
                "--direct_delta_max_damping", str(args.direct_delta_max_damping),
                "--direct_delta_gap_fraction", str(args.direct_delta_gap_fraction),
                "--direct_delta_alpha_max", str(args.direct_delta_alpha_max),
                "--causal_filter_samples", str(args.causal_filter_samples),
                "--causal_filter_min_delta", str(args.causal_filter_min_delta),
            ]
            if args.selection_negative_only:
                cmd.append("--selection_negative_only")
            if args.causal_filter_delta_margin:
                cmd.append("--causal_filter_delta_margin")
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


    selected_domain = domain_configs[args.domain]
    calibration_dataset = selected_domain["calibration_dataset"]
    eval_datasets = selected_domain["eval_datasets"]

    if args.method == 'baseline':
        print(f"\n[DEBUG] ===== BASELINE EVALUATION: domain={args.domain} eval_mode={args.eval_mode} =====")
        evaluate_named_datasets_with_mode(model, tokenizer, eval_datasets, "BASELINE", args.eval_mode)
        raise SystemExit(0)


#     print("\n[DEBUG] ===== COMPUTING EXPERT IMPORTANCE =====")
    expert_importance, mlp_inputs, importance_diagnostics = compute_expert_importance(
        model,
        tokenizer,
        calibration_dataset,
        args.loss_top_k,
        loss_position=args.loss_position,
        loss_type=args.loss_type,
        margin_tau=args.margin_tau,
        lambda_ce=args.lambda_ce,
        boundary_scale=args.boundary_scale,
        boundary_weight_min=args.boundary_weight_min,
        boundary_weight_max=args.boundary_weight_max,
        importance_type=args.importance_type,
        router_reach_temp=args.router_reach_temp,
        router_reach_bias=args.router_reach_bias,
    )

#     print("\n[DEBUG] ===== APPLYING EXPERT BIAS =====")

    # If method is 'default' and loss_top_k is 'top_k', run for all eps values
    if args.method == 'default' and args.loss_top_k == 'top_k':
        eps_values = [0.005, 0.01, 0.05, 0.1, 0.2, 0.3]

        # Compute expert importance once
        print("Computing expert importance (once for all eps values)...")
        expert_importance_computed, mlp_inputs_computed, _importance_diagnostics_computed = compute_expert_importance(
            model,
            tokenizer,
            test_samples_math,
            args.loss_top_k,
            loss_position=args.loss_position,
            loss_type=args.loss_type,
            margin_tau=args.margin_tau,
            lambda_ce=args.lambda_ce,
            boundary_scale=args.boundary_scale,
            boundary_weight_min=args.boundary_weight_min,
            boundary_weight_max=args.boundary_weight_max,
            importance_type=args.importance_type,
            router_reach_temp=args.router_reach_temp,
            router_reach_bias=args.router_reach_bias,
        )

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
            accuracy = test_accuracy(model, tokenizer, test_samples_math, eval_mode=args.eval_mode)
            print(f"AFTER college_mathematics (eps={eps_val}):", accuracy)

            accuracy = test_accuracy(model, tokenizer, test_samples_alg, eval_mode=args.eval_mode)
            print(f"AFTER abstract_algebra (eps={eps_val}):", accuracy)

            accuracy = test_accuracy(model, tokenizer, test_samples_all_part, eval_mode=args.eval_mode)
            print(f"AFTER all_part (eps={eps_val}):", accuracy)
    else:
        # Original single-run logic
        model = run_calibration_method(model, tokenizer, calibration_dataset, expert_importance, mlp_inputs, importance_diagnostics, args)

        # Cleanup to free memory before accuracy evaluation
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()

        # Only test after tuning if calibration was applied
        if args.method != 'baseline':
#         print("\n[DEBUG] ===== TESTING AFTER TUNING =====")
            if getattr(args, "skip_after_evaluation", False):
                print(
                    "[DEBUG] ===== SKIPPING AFTER EVALUATION ===== "
                    "generate_validate selected no candidates, so model bias is unchanged"
                )
            else:
                evaluate_named_datasets_with_mode(model, tokenizer, eval_datasets, "AFTER", args.eval_mode)

            if args.domain == "math" and not getattr(args, "skip_after_evaluation", False):
                accuracy = test_accuracy(model, tokenizer, test_samples_all_part, eval_mode=args.eval_mode)
                print(f"AFTER all_part ({args.eval_mode}):", accuracy)

     #print("[DEBUG] ===== DONE =====")
