import random

import torch
from datasets import load_dataset

from common import dataset_schema
from common.qa_parser import Qwen15QAParser
from common.seeding import set_seed

_DATA_CACHE: dict = {}

# Default QA parser (Qwen1.5). Used when callers don't supply their own.
_DEFAULT_PARSER = Qwen15QAParser()


def _tokenizer_key(tokenizer) -> tuple:
    """Identity of a tokenizer for caching: type + model id."""
    if tokenizer is None:
        return ()
    return (type(tokenizer).__name__, getattr(tokenizer, "name_or_path", None))


def _parser_key(parser) -> tuple:
    """Identity of a QA parser for caching: type + its config attributes."""
    if parser is None:
        return ()
    return (type(parser).__name__, repr(sorted(getattr(parser, "__dict__", {}).items())))


def _cache_key(dataset_kwargs: dict, n: int, seed: int, tokenizer=None, parser=None) -> tuple:
    # Key covers dataset, n, seed, tokenizer, and parser so two models in the
    # same process don't share cached prompts or token ids.
    return (
        tuple(sorted(dataset_kwargs.items())),
        n,
        seed,
        _tokenizer_key(tokenizer),
        _parser_key(parser),
    )


def _join_or(items: list) -> str:
    if len(items) <= 1:
        return "".join(items)
    if len(items) == 2:
        return f"{items[0]} or {items[1]}"
    return ", ".join(items[:-1]) + f", or {items[-1]}"


# Thin wrappers delegating to common.dataset_schema for backward compatibility.
def _extract_question(sample: dict) -> str:
    return dataset_schema.resolve_question(sample)


def _extract_choices(sample: dict) -> tuple:
    return dataset_schema.resolve_choices(sample)


def _extract_answer(sample: dict, labels: list) -> str:
    return dataset_schema.resolve_answer(sample, labels)


# Wrappers that delegate to the default parser for callers that don't inject one.
def _format_prompt(sample: dict, tokenizer, letters: list) -> str:
    return _DEFAULT_PARSER.format_prompt(sample, tokenizer, letters)


def _resolve_answer_token_ids(tokenizer, letters: list) -> list:
    return _DEFAULT_PARSER.resolve_answer_token_ids(tokenizer, letters)


def _prepare(dataset_kwargs: dict, n: int, tokenizer, seed: int, parser=None) -> dict:
    parser = parser if parser is not None else _DEFAULT_PARSER
    full = load_dataset(**dataset_kwargs)
    n = min(n, len(full))
    indices = sorted(random.Random(seed).sample(range(len(full)), n))
    subset = full.select(indices)

    # Everything is resolved per sample: the answer set is NOT assumed uniform
    # across the subset. Homogeneous datasets (MMLU, CommonsenseQA) collapse to
    # the previous behaviour, while ARC-Challenge -- which varies both the choice
    # count (3/4/5) and the label alphabet (A-D vs 1-4) between samples -- no
    # longer aliases every row onto ``subset[0]``'s letters.
    prompts = []
    answer_indices = []
    answer_token_ids = []
    for s in subset:
        letters, _ = dataset_schema.resolve_choices(s)
        prompts.append(parser.format_prompt(s, tokenizer, letters))
        answer_indices.append(dataset_schema.resolve_answer_index(s))
        answer_token_ids.append(parser.resolve_answer_token_ids(tokenizer, letters))

    return {
        "prompts": prompts,
        "answer_indices": answer_indices,
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
    parser=None,
) -> float:
    """Return accuracy in [0, 1] on a multiple-choice QA dataset.

    Uses the injected ``parser`` (defaults to Qwen1.5) for prompt formatting.
    Results are cached per (dataset, n, seed, tokenizer, parser) so two models
    in the same process stay independent. Deterministic with a fixed seed."""
    set_seed(seed)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    parser = parser if parser is not None else _DEFAULT_PARSER
    was_training = model.training
    model.eval()
    try:
        key = _cache_key(dataset_kwargs, n, seed, tokenizer, parser)
        cache = _DATA_CACHE.get(key)
        if cache is None:
            cache = _prepare(dataset_kwargs, n, tokenizer, seed, parser=parser)
            _DATA_CACHE[key] = cache

        prompts = cache["prompts"]
        answer_indices = cache["answer_indices"]
        answer_token_ids = cache["answer_token_ids"]

        correct = 0
        total = 0
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            batch_choice_ids = answer_token_ids[i : i + batch_size]
            batch_truth = answer_indices[i : i + batch_size]

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

            # Per-sample answer sets vary in length (ARC's 3/4/5 choices), so pad
            # each row's candidate-token ids to the batch max and mask the padding
            # to -inf before the argmax. A homogeneous batch has no padding, so the
            # gather/argmax match the previous single-tensor path exactly.
            width = max(len(ids) for ids in batch_choice_ids)
            gather_idx = torch.zeros(
                (len(batch_choice_ids), width), dtype=torch.long, device=model.device
            )
            valid = torch.zeros(
                (len(batch_choice_ids), width), dtype=torch.bool, device=model.device
            )
            for row, ids in enumerate(batch_choice_ids):
                gather_idx[row, : len(ids)] = torch.tensor(ids, device=model.device)
                valid[row, : len(ids)] = True

            answer_logits = torch.gather(last_logits, 1, gather_idx)
            answer_logits = answer_logits.masked_fill(~valid, float("-inf"))
            predicted = torch.argmax(answer_logits, dim=1)
            truth = torch.tensor(batch_truth, device=model.device)

            correct += (predicted == truth).sum().item()
            total += len(batch_prompts)

            del inputs, outputs, last_logits, gather_idx, valid
            del answer_logits, predicted, truth
            torch.cuda.empty_cache()

        return correct / total
    finally:
        if was_training:
            model.train()
