"""Tests for the declarative dataset-schema registry (S7, §4.2/§3.2).

Covers the S7 gate for the schema resolvers:

* **parity with the inline ``qa_eval._extract_*`` helpers** (the equivalence the
  golden gate protects): for a representative sample of every shape (MMLU
  int-answer, MMLU list-choices, ARC/OpenBookQA dict-choices + answerKey,
  Winogrande option1/2 + numeric-string answer, CommonsenseQA) the module-level
  ``resolve_question`` / ``resolve_choices`` / ``resolve_answer`` return values
  identical to ``_extract_question`` / ``_extract_choices`` / ``_extract_answer``;
* the per-field key-presence priority (question ``question`` > ``sentence`` >
  ``question_stem``; choices dict > list > options; answer ``answerKey`` >
  ``answer``);
* the named :data:`SCHEMA_REGISTRY` table + :func:`detect_schema`, and that each
  detected ``DatasetSchema`` resolves its family identically to the inline code.
"""

import pytest

from common import dataset_schema, qa_eval
from common.dataset_schema import (
    SCHEMA_REGISTRY,
    DatasetSchema,
    detect_schema,
    resolve_answer,
    resolve_answer_index,
    resolve_choices,
    resolve_question,
)
from common.registry import Registry


# Representative sample of every shape the pipeline feeds the resolvers, plus the
# expected schema name :func:`detect_schema` should return for it.
SAMPLES = {
    "mmlu_int": (
        {"question": "Q1?", "choices": ["a", "b", "c", "d"], "answer": 2},
        "mmlu",
    ),
    "mmlu_list": (
        {"question": "Q2?", "choices": ["x", "y"], "answer": 0},
        "mmlu",
    ),
    "arc": (
        {
            "question": "Q3?",
            "choices": {"label": ["A", "B", "C", "D"], "text": ["a", "b", "c", "d"]},
            "answerKey": "B",
        },
        "commonsense_qa",  # ARC is key-identical to CommonsenseQA
    ),
    "openbookqa": (
        {
            "question_stem": "stem?",
            "choices": {"label": ["A", "B", "C"], "text": ["x", "y", "z"]},
            "answerKey": "C",
        },
        "openbookqa",
    ),
    "winogrande": (
        {"sentence": "The _ is big.", "option1": "cat", "option2": "dog", "answer": "2"},
        "winogrande",
    ),
    "commonsense_qa": (
        {
            "question": "CQ?",
            "choices": {
                "label": ["A", "B", "C", "D", "E"],
                "text": ["a", "b", "c", "d", "e"],
            },
            "answerKey": "D",
        },
        "commonsense_qa",
    ),
}


def _sample(name):
    return SAMPLES[name][0]


# --------------------------------------------------------------------------- #
# Parity: module-level resolvers == inline qa_eval._extract_* (every shape).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", list(SAMPLES))
def test_resolve_question_matches_inline(name):
    sample = _sample(name)
    assert resolve_question(sample) == qa_eval._extract_question(sample)


@pytest.mark.parametrize("name", list(SAMPLES))
def test_resolve_choices_matches_inline(name):
    sample = _sample(name)
    assert resolve_choices(sample) == qa_eval._extract_choices(sample)


@pytest.mark.parametrize("name", list(SAMPLES))
def test_resolve_answer_matches_inline(name):
    sample = _sample(name)
    labels, _ = qa_eval._extract_choices(sample)
    assert resolve_answer(sample, labels) == qa_eval._extract_answer(sample, labels)


# --------------------------------------------------------------------------- #
# Parity through the detected named schema.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", list(SAMPLES))
def test_detected_schema_resolves_identically_to_inline(name):
    sample = _sample(name)
    schema = detect_schema(sample)
    labels, texts = qa_eval._extract_choices(sample)

    assert schema.resolve_question(sample) == qa_eval._extract_question(sample)
    assert schema.resolve_choices(sample) == (labels, texts)
    assert schema.resolve_answer(sample, labels) == qa_eval._extract_answer(sample, labels)


@pytest.mark.parametrize("name", list(SAMPLES))
def test_detect_schema_returns_expected_name(name):
    sample, expected = SAMPLES[name]
    assert detect_schema(sample).name == expected


# --------------------------------------------------------------------------- #
# Per-field key-presence priority (mirrors the inline if/elif order).
# --------------------------------------------------------------------------- #
def test_resolve_question_priority():
    assert resolve_question({"question": "q"}) == "q"
    assert resolve_question({"sentence": "s"}) == "s"
    assert resolve_question({"question_stem": "qs"}) == "qs"
    # 'question' wins over the lower-priority keys when several are present.
    assert resolve_question({"question": "q", "sentence": "s", "question_stem": "qs"}) == "q"
    assert resolve_question({"sentence": "s", "question_stem": "qs"}) == "s"


def test_resolve_question_unknown_raises():
    with pytest.raises(KeyError):
        resolve_question({"nope": 1})


def test_resolve_choices_dict_form():
    labels, texts = resolve_choices({"choices": {"label": ["A", "B"], "text": ["x", "y"]}})
    assert labels == ["A", "B"]
    assert texts == ["x", "y"]


def test_resolve_choices_list_form_generates_letters():
    labels, texts = resolve_choices({"choices": ["x", "y", "z"]})
    assert labels == ["A", "B", "C"]
    assert texts == ["x", "y", "z"]


def test_resolve_choices_options_form():
    labels, texts = resolve_choices({"option1": "first", "option2": "second"})
    assert labels == ["A", "B"]
    assert texts == ["first", "second"]


def test_resolve_choices_choices_wins_over_options():
    # 'choices' present takes priority over option1/option2 (== inline order).
    sample = {"choices": ["x", "y"], "option1": "p", "option2": "q"}
    assert resolve_choices(sample) == (["A", "B"], ["x", "y"])


def test_resolve_choices_unknown_raises():
    with pytest.raises(KeyError):
        resolve_choices({"foo": "bar"})


def test_resolve_answer_answer_key():
    assert resolve_answer({"answerKey": "C"}, ["A", "B", "C"]) == "C"


def test_resolve_answer_integer_index():
    assert resolve_answer({"answer": 2}, ["A", "B", "C", "D"]) == "C"


def test_resolve_answer_already_letter():
    assert resolve_answer({"answer": "B"}, ["A", "B", "C"]) == "B"


def test_resolve_answer_numeric_string():
    assert resolve_answer({"answer": "1"}, ["A", "B"]) == "A"
    assert resolve_answer({"answer": "2"}, ["A", "B"]) == "B"


def test_resolve_answer_answer_key_wins_over_answer():
    # 'answerKey' present is read before the 'answer' value path (== inline order).
    assert resolve_answer({"answerKey": "C", "answer": 0}, ["A", "B", "C"]) == "C"


def test_resolve_answer_missing_raises():
    with pytest.raises(KeyError):
        resolve_answer({"foo": "bar"}, ["A", "B"])


# --------------------------------------------------------------------------- #
# resolve_answer_index: per-sample 0-based index, robust to heterogeneous labels.
# --------------------------------------------------------------------------- #
def test_resolve_answer_index_mmlu_integer():
    sample = {"question": "q", "choices": ["w", "x", "y", "z"], "answer": 2}
    assert resolve_answer_index(sample) == 2


def test_resolve_answer_index_lettered_answer_key():
    sample = {
        "question": "q",
        "choices": {"label": ["A", "B", "C", "D"], "text": ["a", "b", "c", "d"]},
        "answerKey": "C",
    }
    assert resolve_answer_index(sample) == 2


def test_resolve_answer_index_numeric_labels():
    # ARC's occasional numeric alphabet: '3' is the third of ['1','2','3','4'].
    sample = {
        "question": "q",
        "choices": {"label": ["1", "2", "3", "4"], "text": ["a", "b", "c", "d"]},
        "answerKey": "3",
    }
    assert resolve_answer_index(sample) == 2


def test_resolve_answer_index_five_choices():
    sample = {
        "question": "q",
        "choices": {"label": ["A", "B", "C", "D", "E"],
                    "text": ["a", "b", "c", "d", "e"]},
        "answerKey": "E",
    }
    assert resolve_answer_index(sample) == 4


def test_resolve_answer_index_three_choices():
    sample = {
        "question": "q",
        "choices": {"label": ["A", "B", "C"], "text": ["a", "b", "c"]},
        "answerKey": "A",
    }
    assert resolve_answer_index(sample) == 0


def test_resolve_answer_index_marker_format_mismatch():
    # Defensive fallback: answerKey format disagrees with the sample's labels.
    # Numeric answerKey against lettered labels -> 1-based -> index.
    lettered = {
        "question": "q",
        "choices": {"label": ["A", "B", "C", "D"], "text": ["a", "b", "c", "d"]},
        "answerKey": "2",
    }
    assert resolve_answer_index(lettered) == 1
    # Lettered answerKey against numeric labels -> A-based -> index.
    numeric = {
        "question": "q",
        "choices": {"label": ["1", "2", "3", "4"], "text": ["a", "b", "c", "d"]},
        "answerKey": "B",
    }
    assert resolve_answer_index(numeric) == 1


def test_resolve_answer_index_lowercase_answer_key_normalizes_case():
    # Defensive: a lowercase marker is upper-cased before mapping, so 'c' against
    # A-D labels resolves to index 2 rather than raising.
    sample = {
        "question": "q",
        "choices": {"label": ["A", "B", "C", "D"], "text": ["a", "b", "c", "d"]},
        "answerKey": "c",
    }
    assert resolve_answer_index(sample) == 2


def test_resolve_answer_index_out_of_range_raises():
    # 'E' would be index 4, but the item only offers three choices.
    sample = {
        "question": "q",
        "choices": {"label": ["A", "B", "C"], "text": ["a", "b", "c"]},
        "answerKey": "E",
    }
    with pytest.raises(KeyError):
        resolve_answer_index(sample)


# --------------------------------------------------------------------------- #
# Registry shape and descriptor.
# --------------------------------------------------------------------------- #
def test_schema_registry_is_a_registry_with_named_schemas():
    assert isinstance(SCHEMA_REGISTRY, Registry)
    assert SCHEMA_REGISTRY.names() == ["mmlu", "winogrande", "commonsense_qa", "openbookqa"]
    for name in SCHEMA_REGISTRY.names():
        schema = SCHEMA_REGISTRY[name]
        assert isinstance(schema, DatasetSchema)
        assert schema.name == name


def test_dataset_schema_is_frozen_dataclass():
    schema = SCHEMA_REGISTRY["mmlu"]
    with pytest.raises((AttributeError, TypeError)):
        schema.name = "other"


def test_detect_schema_unknown_raises():
    with pytest.raises(KeyError):
        detect_schema({"totally": "unknown"})
