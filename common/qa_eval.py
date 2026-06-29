import random

import torch
from datasets import load_dataset

from common import dataset_schema
from common.qa_parser import Qwen15QAParser
from common.seeding import set_seed

_DATA_CACHE: dict = {}

# Default model-family QA parser. The prompt format and answer-token resolution
# are owned by a Qwen15QAParser; the module-level default reproduces the inline
# behaviour so callers that pass no parser get the same result.
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
    # The cache stores tokenizer-rendered prompts and encoded answer ids, so the
    # key must include everything they depend on: the dataset, n, the ``seed``
    # the subset is drawn with (``random.Random(seed)`` -- otherwise the first
    # seed silently wins), and the tokenizer + parser identity (otherwise a second
    # model in the same process would reuse the first model's prompts/token ids).
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


# The schema-aware field extraction lives in the declarative
# common.dataset_schema registry; these stay as thin, behaviour-identical
# wrappers so existing callers/tests keep working.
def _extract_question(sample: dict) -> str:
    return dataset_schema.resolve_question(sample)


def _extract_choices(sample: dict) -> tuple:
    return dataset_schema.resolve_choices(sample)


def _extract_answer(sample: dict, labels: list) -> str:
    return dataset_schema.resolve_answer(sample, labels)


# Prompt format and answer-token resolution are owned by the model-family
# parser; these wrappers delegate to the module default so the inline
# behaviour is preserved for callers that do not inject their own parser.
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

    answer_letters, _ = dataset_schema.resolve_choices(subset[0])
    letter_to_idx = {letter: i for i, letter in enumerate(answer_letters)}

    prompts = [parser.format_prompt(s, tokenizer, answer_letters) for s in subset]
    ground_truths = [dataset_schema.resolve_answer(s, answer_letters) for s in subset]
    answer_token_ids = parser.resolve_answer_token_ids(tokenizer, answer_letters)

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
    parser=None,
) -> float:
    """Evaluate accuracy (number from 0 to 1) of a model in a given question-answer dataset.

    ``parser`` is the injected model-family QA parser; when unset the module
    default :data:`_DEFAULT_PARSER` (Qwen1.5) is used, reproducing the inline
    prompt/answer-token behaviour. The tokenizer and parser identity are
    part of the subset cache key, so evaluating a second model in the same process
    does not reuse the first model's rendered prompts.

    Result is constant with a fixed seed."""
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
