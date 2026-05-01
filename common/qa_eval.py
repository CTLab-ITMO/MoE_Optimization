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


def _format_prompt(sample: dict, tokenizer, letters: list) -> str:
    options_text = ""
    for label, text in zip(sample["choices"]["label"], sample["choices"]["text"]):
        options_text += f"{label}. {text}\n"

    system_message = (
        "You are a precise evaluation system. "
        "You will be given a question and multiple-choice options. "
        f"Your task is to answer with ONLY the letter corresponding to the correct option ({_join_or(letters)}). "
        "Do not write the full answer, do not write explanations. Output only one character."
    )
    user_message = f"Question: {sample['question']}\nOptions:\n{options_text}\nAnswer:"
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
        toks = tokenizer.encode(f" {letter}", add_special_tokens=False)
        if not toks:
            toks = tokenizer.encode(letter, add_special_tokens=False)
        ids.append(toks[0] if toks else -1)
    return ids


def _prepare(dataset_kwargs: dict, n: int, tokenizer, seed: int) -> dict:
    full = load_dataset(**dataset_kwargs)
    n = min(n, len(full))
    indices = sorted(random.Random(seed).sample(range(len(full)), n))
    subset = full.select(indices)

    # Per-dataset letter set: assume every sample shares the schema of subset[0].
    answer_letters = list(subset[0]["choices"]["label"])
    letter_to_idx = {letter: i for i, letter in enumerate(answer_letters)}

    prompts = [_format_prompt(s, tokenizer, answer_letters) for s in subset]
    ground_truths = [s["answerKey"] for s in subset]
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
    batch_size: int = 100,
    max_length: int = 512,
) -> float:
    set_seed(seed)
    tokenizer.padding_side = "left"

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
