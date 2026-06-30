"""Declarative schema registry for QA evaluation samples.

Handles five dataset shapes (MMLU, Winogrande, ARC, OpenBookQA, CommonsenseQA)
via ordered, presence-keyed rule tables instead of if/elif chains.
Each field (question, choices, answer) has its own rules; the first matching
rule wins, so hybrids work without guessing the whole-sample schema.

Public API: ``resolve_question``, ``resolve_choices``, ``resolve_answer``.
Named per-family schemas live in ``SCHEMA_REGISTRY``; ``detect_schema`` picks one.

Library code only -- never imports an experiment config.
"""

from dataclasses import dataclass
from typing import Callable

from common.registry import Registry

# Question-key lookup order: "question" (MMLU/ARC/CommonsenseQA) -> "sentence" (Winogrande)
# -> "question_stem" (OpenBookQA). First key found wins.
QUESTION_KEYS: tuple[str, ...] = ("question", "sentence", "question_stem")


# --------------------------------------------------------------------------- #
# Field strategy functions used by both the rule tables and SCHEMA_REGISTRY.
# --------------------------------------------------------------------------- #
def _choices_from_dict(sample: dict) -> tuple[list, list]:
    """``choices`` is a ``{"label": [...], "text": [...]}`` dict (ARC / OpenBookQA / CommonsenseQA)."""
    choices = sample["choices"]
    return list(choices["label"]), list(choices["text"])


def _choices_from_list(sample: dict) -> tuple[list, list]:
    """``choices`` is a plain list; synthesize ``A, B, C, ...`` labels (MMLU)."""
    choices = sample["choices"]
    labels = [chr(ord("A") + i) for i in range(len(choices))]
    return labels, list(choices)


def _choices_from_options(sample: dict) -> tuple[list, list]:
    """``option1`` / ``option2`` with fixed ``["A", "B"]`` labels (Winogrande)."""
    return ["A", "B"], [sample["option1"], sample["option2"]]


def _resolve_answer_value(ans, labels: list) -> str:
    """Normalize the ``"answer"`` field to a label letter.

    int -> labels[ans] (MMLU); letter already in labels -> returned as-is;
    1-based string -> labels[int(ans)-1] (Winogrande: '1'->'A', '2'->'B').
    """
    if isinstance(ans, int):  # mmlu
        return labels[ans]
    if ans in labels:  # already a letter
        return ans
    return labels[int(ans) - 1]  # winogrande: '1' -> 'A', '2' -> 'B'


# --------------------------------------------------------------------------- #
# Per-field rule tables: ordered, presence-keyed; first match wins.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _ChoiceRule:
    detect: Callable[[dict], bool]
    resolve: Callable[[dict], tuple[list, list]]


@dataclass(frozen=True)
class _AnswerRule:
    detect: Callable[[dict], bool]
    resolve: Callable[[dict, list], str]


_CHOICE_RULES: tuple[_ChoiceRule, ...] = (
    _ChoiceRule(lambda s: "choices" in s and isinstance(s["choices"], dict), _choices_from_dict),
    _ChoiceRule(lambda s: "choices" in s and not isinstance(s["choices"], dict), _choices_from_list),
    _ChoiceRule(lambda s: "option1" in s and "option2" in s, _choices_from_options),
)

_ANSWER_RULES: tuple[_AnswerRule, ...] = (
    _AnswerRule(lambda s: "answerKey" in s, lambda s, labels: s["answerKey"]),
    _AnswerRule(lambda s: "answer" in s, lambda s, labels: _resolve_answer_value(s["answer"], labels)),
)


# --------------------------------------------------------------------------- #
# Public resolvers driven by the rule tables above; work for any sample shape.
# --------------------------------------------------------------------------- #
def resolve_question(sample: dict) -> str:
    """Return question text using key-presence priority order."""
    for key in QUESTION_KEYS:
        if key in sample:
            return sample[key]
    raise KeyError(f"sample has none of the question keys {QUESTION_KEYS}")


def resolve_choices(sample: dict) -> tuple[list, list]:
    """Return ``(labels, texts)`` for a sample's choices field."""
    for rule in _CHOICE_RULES:
        if rule.detect(sample):
            return rule.resolve(sample)
    raise KeyError("sample has no recognized choices field")


def resolve_answer(sample: dict, labels: list) -> str:
    """Return the ground-truth answer letter for a sample."""
    for rule in _ANSWER_RULES:
        if rule.detect(sample):
            return rule.resolve(sample, labels)
    # Raise KeyError when neither "answerKey" nor "answer" is present.
    raise KeyError("answer")


# --------------------------------------------------------------------------- #
# Declarative named schemas: one descriptor per dataset family.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DatasetSchema:
    """Descriptor for one dataset family's sample shape.

    Holds the question key, choices resolver, answer strategy, and a ``detect``
    predicate. Registered by name in ``SCHEMA_REGISTRY``; ``detect_schema``
    returns the first schema whose ``detect`` matches.
    """

    name: str
    detect: Callable[[dict], bool]
    question_key: str
    choices_resolver: Callable[[dict], tuple[list, list]]
    answer_uses_key: bool  # True -> read 'answerKey'; False -> the 'answer' value path

    def resolve_question(self, sample: dict) -> str:
        return sample[self.question_key]

    def resolve_choices(self, sample: dict) -> tuple[list, list]:
        return self.choices_resolver(sample)

    def resolve_answer(self, sample: dict, labels: list) -> str:
        if self.answer_uses_key:
            return sample["answerKey"]
        return _resolve_answer_value(sample["answer"], labels)


def _build_schema_registry() -> Registry:
    """Build and return the named schema registry in detection-priority order.

    ``commonsense_qa`` is registered before ``openbookqa`` so ``"question"``
    takes priority over ``"question_stem"`` for hybrid samples. It also covers
    ARC, which has the same key signature.
    """
    registry = Registry("dataset_schema")
    registry.register(
        "mmlu",
        DatasetSchema(
            name="mmlu",
            detect=lambda s: "choices" in s and not isinstance(s["choices"], dict),
            question_key="question",
            choices_resolver=_choices_from_list,
            answer_uses_key=False,
        ),
    )
    registry.register(
        "winogrande",
        DatasetSchema(
            name="winogrande",
            detect=lambda s: "option1" in s and "option2" in s,
            question_key="sentence",
            choices_resolver=_choices_from_options,
            answer_uses_key=False,
        ),
    )
    registry.register(
        "commonsense_qa",  # also covers ARC (question + dict choices + answerKey)
        DatasetSchema(
            name="commonsense_qa",
            detect=lambda s: "choices" in s and isinstance(s["choices"], dict) and "question" in s,
            question_key="question",
            choices_resolver=_choices_from_dict,
            answer_uses_key=True,
        ),
    )
    registry.register(
        "openbookqa",
        DatasetSchema(
            name="openbookqa",
            detect=lambda s: "choices" in s and isinstance(s["choices"], dict) and "question_stem" in s,
            question_key="question_stem",
            choices_resolver=_choices_from_dict,
            answer_uses_key=True,
        ),
    )
    return registry


# Registry mapping name -> DatasetSchema; extend by registering a new schema in-tree.
SCHEMA_REGISTRY: Registry = _build_schema_registry()


def detect_schema(sample: dict) -> DatasetSchema:
    """Return the first ``DatasetSchema`` whose ``detect`` matches, or raise ``KeyError``."""
    for name in SCHEMA_REGISTRY.names():
        schema: DatasetSchema = SCHEMA_REGISTRY[name]
        if schema.detect(sample):
            return schema
    raise KeyError(f"sample matches no registered DatasetSchema; keys={sorted(sample)}")
