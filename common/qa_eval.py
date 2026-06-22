import random

import torch
from datasets import load_dataset

from common.seeding import set_seed

_DATA_CACHE: dict = {}


def _cache_key(dataset_kwargs: dict, n: int) -> tuple:
    return (tuple(sorted(dataset_kwargs.items())), n)


def _join_or(items: list) -> str:
    if len(items) <= 1:
        return "".join(items)
    if len(items) == 2:
        return f"{items[0]} or {items[1]}"
    return ", ".join(items[:-1]) + f", or {items[-1]}"


def _extract_question(sample: dict) -> str:
    if "question" in sample:
        return sample["question"]
    if "sentence" in sample:  # winogrande
        return sample["sentence"]
    if "question_stem" in sample:  # openbookqa
        return sample["question_stem"]
    raise KeyError("sample has neither 'question' nor 'sentence'")


def _extract_choices(sample: dict) -> tuple:
    if "choices" in sample:
        choices = sample["choices"]
        if isinstance(choices, dict):  # commonsense_qa, ARC, OpenBookQA
            return list(choices["label"]), list(choices["text"])
        labels = [chr(ord("A") + i) for i in range(len(choices))]  # mmlu
        return labels, list(choices)
    if "option1" in sample and "option2" in sample:  # winogrande
        return ["A", "B"], [sample["option1"], sample["option2"]]
    raise KeyError("sample has no recognized choices field")


def _extract_answer(sample: dict, labels: list) -> str:
    if "answerKey" in sample:
        return sample["answerKey"]
    ans = sample["answer"]
    if isinstance(ans, int):  # mmlu
        return labels[ans]
    if ans in labels:  # already a letter
        return ans
    return labels[int(ans) - 1]  # winogrande: '1' -> 'A', '2' -> 'B'


def _format_prompt(sample: dict, tokenizer, letters: list) -> str:
    labels, texts = _extract_choices(sample)
    options_text = ""
    for label, text in zip(labels, texts):
        options_text += f"{label}. {text}\n"

    system_message = (
        "You are a precise evaluation system. "
        "You will be given a question and multiple-choice options. "
        f"Your task is to answer with ONLY the letter corresponding to the correct option ({_join_or(letters)}). "
        "Do not write the full answer, do not write explanations. Output only one character."
    )
    user_message = f"Question: {_extract_question(sample)}\nOptions:\n{options_text}\nAnswer:"
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _resolve_answer_token_ids(tokenizer, letters: list) -> list:
    ids = []
    for letter in letters:
        toks = tokenizer.encode(letter, add_special_tokens=False)
        ids.append(toks[0])
    return ids


def _prepare(dataset_kwargs: dict, n: int, tokenizer, seed: int) -> dict:
    full = load_dataset(**dataset_kwargs)
    n = min(n, len(full))
    indices = sorted(random.Random(seed).sample(range(len(full)), n))
    subset = full.select(indices)

    answer_letters, _ = _extract_choices(subset[0])
    letter_to_idx = {letter: i for i, letter in enumerate(answer_letters)}

    prompts = [_format_prompt(s, tokenizer, answer_letters) for s in subset]
    ground_truths = [_extract_answer(s, answer_letters) for s in subset]
    answer_token_ids = _resolve_answer_token_ids(tokenizer, answer_letters)

    return {
        "prompts": prompts,
        "ground_truths": ground_truths,
        "answer_letters": answer_letters,
        "letter_to_idx": letter_to_idx,
        "answer_token_ids": answer_token_ids,
    }


@torch.no_grad()
def evaluate_qa(
    model,
    tokenizer,
    dataset_kwargs: dict,
    n: int,
    *,
    seed: int = 42,
    batch_size: int = 16,
    max_length: int = 512,
) -> float:
    """Evaluate accuracy (number from 0 to 1) of a model in a given question-answer dataset.
    
    Result is constant with a fixed seed."""
    set_seed(seed)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    was_training = model.training
    model.eval()
    try:
        key = _cache_key(dataset_kwargs, n)
        cache = _DATA_CACHE.get(key)
        if cache is None:
            cache = _prepare(dataset_kwargs, n, tokenizer, seed)
            _DATA_CACHE[key] = cache

        prompts = cache["prompts"]
        ground_truths = cache["ground_truths"]
        letter_to_idx = cache["letter_to_idx"]
        answer_token_tensor = torch.tensor(
            cache["answer_token_ids"], device=model.device
        )

        correct = 0
        total = 0
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            batch_truths = ground_truths[i : i + batch_size]

            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(model.device)

            outputs = model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                output_hidden_states=False,
                output_attentions=False,
                use_cache=False,
            )
            last_logits = outputs.logits[:, -1, :]

            idx = answer_token_tensor.unsqueeze(0).expand(last_logits.size(0), -1)
            answer_logits = torch.gather(last_logits, 1, idx)
            predicted = torch.argmax(answer_logits, dim=1)
            truth = torch.tensor(
                [letter_to_idx[t] for t in batch_truths], device=model.device
            )

            correct += (predicted == truth).sum().item()
            total += len(batch_prompts)

            del inputs, outputs, last_logits, idx, answer_logits, predicted, truth
            torch.cuda.empty_cache()

        return correct / total
    finally:
        if was_training:
            model.train()
