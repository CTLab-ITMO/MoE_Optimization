"""Declarative dataset-schema registry for QA evaluation samples.

QA evaluation samples arrive in several shapes (MMLU, Winogrande, ARC,
OpenBookQA, CommonsenseQA). Rather than reading each shape through a
hand-written ``if/elif`` chain in three helpers:

* ``_extract_question`` -- ``"question"`` else ``"sentence"`` else ``"question_stem"``;
* ``_extract_choices`` -- a ``"choices"`` dict (label/text), a ``"choices"`` list
  (synthesize ``A.. letters``), or ``"option1"/"option2"``;
* ``_extract_answer`` -- ``"answerKey"``, else the ``"answer"`` value (an int
  index, an already-resolved letter, or a 1-based numeric string),

this module uses a *declarative table*: each field has an ordered tuple of
presence-keyed rules whose first match wins, plus a registry of named
:class:`DatasetSchema` descriptors -- one per dataset family -- usable for
introspection / detection. The module-level :func:`resolve_question`,
:func:`resolve_choices` and :func:`resolve_answer` handle every sample shape,
including hybrids, because they are driven by per-field key-presence priority
rather than by a whole-sample schema guess.

Library code only -- this module never imports an experiment ``config``.
"""

from dataclasses import dataclass
from typing import Callable

from common.registry import Registry

# Entry-point group a third-party package uses to advertise a dataset-schema
# plugin (pulled in only on an explicit ``SCHEMA_REGISTRY.load_entry_points``
# call -- the canonical registry stays pinned in-tree; see :mod:`common.registry`).
SCHEMA_ENTRY_POINT_GROUP = "moe.dataset_schema"

# Ordered question-key priority, mirroring ``qa_eval._extract_question`` exactly:
# ``question`` (MMLU / ARC / CommonsenseQA) -> ``sentence`` (Winogrande) ->
# ``question_stem`` (OpenBookQA).
QUESTION_KEYS: tuple[str, ...] = ("question", "sentence", "question_stem")


# --------------------------------------------------------------------------- #
# Field strategy functions (declared once; referenced by both the per-field
# rule tables below and the named schemas in SCHEMA_REGISTRY).
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
    """Resolve the ``"answer"`` field value, mirroring ``qa_eval._extract_answer``.

    ``int`` -> ``labels[ans]`` (MMLU); an already-resolved letter is returned
    as-is; otherwise a 1-based numeric string -> ``labels[int(ans) - 1]``
    (Winogrande: ``'1'`` -> ``'A'``, ``'2'`` -> ``'B'``).
    """
    if isinstance(ans, int):  # mmlu
        return labels[ans]
    if ans in labels:  # already a letter
        return ans
    return labels[int(ans) - 1]  # winogrande: '1' -> 'A', '2' -> 'B'


# --------------------------------------------------------------------------- #
# Per-field rule tables: ordered, presence-keyed; first match wins. The order
# IS the ``if/elif`` priority of the inline ``_extract_*`` helpers.
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
# Module-level resolvers (the canonical path the stages wire against). Driven by
# the per-field rule tables, so behaviour is identical to ``_extract_*`` for any
# sample shape -- no whole-sample schema guess that could diverge on a hybrid.
# --------------------------------------------------------------------------- #
def resolve_question(sample: dict) -> str:
    """Question text, by key-presence priority (== ``qa_eval._extract_question``)."""
    for key in QUESTION_KEYS:
        if key in sample:
            return sample[key]
    raise KeyError("sample has neither 'question' nor 'sentence'")


def resolve_choices(sample: dict) -> tuple[list, list]:
    """``(labels, texts)`` for a sample (== ``qa_eval._extract_choices``)."""
    for rule in _CHOICE_RULES:
        if rule.detect(sample):
            return rule.resolve(sample)
    raise KeyError("sample has no recognized choices field")


def resolve_answer(sample: dict, labels: list) -> str:
    """Ground-truth answer letter for a sample (== ``qa_eval._extract_answer``)."""
    for rule in _ANSWER_RULES:
        if rule.detect(sample):
            return rule.resolve(sample, labels)
    # Mirror the ``KeyError`` the inline code raises via ``sample["answer"]``
    # when neither ``answerKey`` nor ``answer`` is present.
    raise KeyError("answer")


# --------------------------------------------------------------------------- #
# Declarative named schemas: one descriptor per dataset family.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DatasetSchema:
    """Declarative descriptor of one dataset family's sample shape.

    Bundles the three field resolvers (question key, choices strategy, answer
    strategy) and a ``detect`` predicate keyed on which keys a sample carries.
    Registered by name in :data:`SCHEMA_REGISTRY`; :func:`detect_schema` returns
    the first schema (in registration order) whose ``detect`` matches. The
    bundled :meth:`resolve_question` / :meth:`resolve_choices` /
    :meth:`resolve_answer` reproduce the inline ``qa_eval._extract_*`` behaviour
    for that family.
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
    """The canonical dataset-schema registry, in detection-priority order.

    ``commonsense_qa`` is registered before ``openbookqa`` so that the question
    key priority (``"question"`` over ``"question_stem"``, matching
    :data:`QUESTION_KEYS`) is preserved for any hybrid sample carrying both keys.
    The ``commonsense_qa`` schema (``question`` + dict ``choices`` + ``answerKey``)
    also covers ARC, which is key-identical.
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


# The single source of truth for named dataset schemas: a
# name -> DatasetSchema Mapping, extensible via ``register`` / ``load_entry_points``.
SCHEMA_REGISTRY: Registry = _build_schema_registry()


def detect_schema(sample: dict) -> DatasetSchema:
    """Return the first registered :class:`DatasetSchema` whose ``detect`` matches.

    Raises :class:`KeyError` if no schema recognizes the sample's keys.
    """
    for name in SCHEMA_REGISTRY.names():
        schema: DatasetSchema = SCHEMA_REGISTRY[name]
        if schema.detect(sample):
            return schema
    raise KeyError(f"sample matches no registered DatasetSchema; keys={sorted(sample)}")
