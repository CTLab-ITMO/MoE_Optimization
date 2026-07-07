"""Tests for the QA prompt / answer-token parser (S7, §3.2/§3.7).

Covers the S7 gate for the parser adapter:

* **parity with the inline ``qa_eval`` helpers** (the equivalence the golden gate
  protects): for a representative sample of every shape, ``Qwen15QAParser``
  default behaviour reproduces ``qa_eval._format_prompt`` and
  ``qa_eval._resolve_answer_token_ids`` exactly;
* the §3.2 "leading space" subtlety is a constructor flag whose DEFAULT
  (``leading_space=False``) encodes the bare letter (current behaviour), while
  ``True`` encodes the leading-space variant;
* ``_join_or`` is reproduced (the valid-letter enumeration in the prompt);
* §3.7 housekeeping: the abstract base cannot be instantiated and the concrete
  ``format_prompt`` / ``resolve_answer_token_ids`` reject an unknown keyword.
"""

import pytest

from common import qa_eval
from common.qa_parser import MixtralQAParser, QAParser, Qwen15QAParser, _join_or
from common.testing.fakes import FakeTokenizer


# Representative sample of every shape (mirrors test_dataset_schema's table).
SAMPLES = {
    "mmlu_int": {"question": "Q1?", "choices": ["a", "b", "c", "d"], "answer": 2},
    "mmlu_list": {"question": "Q2?", "choices": ["x", "y"], "answer": 0},
    "arc": {
        "question": "Q3?",
        "choices": {"label": ["A", "B", "C", "D"], "text": ["a", "b", "c", "d"]},
        "answerKey": "B",
    },
    "openbookqa": {
        "question_stem": "stem?",
        "choices": {"label": ["A", "B", "C"], "text": ["x", "y", "z"]},
        "answerKey": "C",
    },
    "winogrande": {
        "sentence": "The _ is big.",
        "option1": "cat",
        "option2": "dog",
        "answer": "2",
    },
    "commonsense_qa": {
        "question": "CQ?",
        "choices": {"label": ["A", "B", "C", "D", "E"], "text": ["a", "b", "c", "d", "e"]},
        "answerKey": "D",
    },
}


def _letters(sample):
    labels, _ = qa_eval._extract_choices(sample)
    return labels


# --------------------------------------------------------------------------- #
# Parity: format_prompt == qa_eval._format_prompt (every shape).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", list(SAMPLES))
def test_format_prompt_matches_inline(name):
    sample = SAMPLES[name]
    tok = FakeTokenizer()
    letters = _letters(sample)
    parser = Qwen15QAParser()

    produced = parser.format_prompt(sample, tok, letters)
    expected = qa_eval._format_prompt(sample, FakeTokenizer(), letters)
    assert produced == expected


def test_format_prompt_includes_question_and_options():
    sample = SAMPLES["mmlu_int"]
    prompt = Qwen15QAParser().format_prompt(sample, FakeTokenizer(), _letters(sample))
    assert "Q1?" in prompt
    assert "A. a" in prompt
    assert "A, B, C, or D" in prompt  # _join_or over the four letters


# --------------------------------------------------------------------------- #
# Parity: resolve_answer_token_ids default == qa_eval._resolve_answer_token_ids.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("letters", [["A", "B"], ["A", "B", "C", "D"], ["A", "B", "C", "D", "E"]])
def test_resolve_answer_token_ids_matches_inline(letters):
    tok = FakeTokenizer()
    produced = Qwen15QAParser().resolve_answer_token_ids(tok, letters)
    expected = qa_eval._resolve_answer_token_ids(FakeTokenizer(), letters)
    assert produced == expected


def test_default_leading_space_is_false_and_encodes_bare_letter():
    parser = Qwen15QAParser()
    assert parser.leading_space is False
    # FakeTokenizer.encode returns [ord(text[0])]: bare letter -> ord(letter).
    ids = parser.resolve_answer_token_ids(FakeTokenizer(), ["A", "B"])
    assert ids == [ord("A"), ord("B")]


def test_leading_space_true_encodes_with_space_prefix():
    parser = Qwen15QAParser(leading_space=True)
    # With a leading space the first char is " " -> ord(" ").
    ids = parser.resolve_answer_token_ids(FakeTokenizer(), ["A", "B"])
    assert ids == [ord(" "), ord(" ")]
    # And it differs from the default (bare-letter) encoding.
    assert ids != Qwen15QAParser().resolve_answer_token_ids(FakeTokenizer(), ["A", "B"])


# --------------------------------------------------------------------------- #
# _join_or reproduction.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "items, expected",
    [
        ([], ""),
        (["A"], "A"),
        (["A", "B"], "A or B"),
        (["A", "B", "C"], "A, B, or C"),
    ],
)
def test_join_or(items, expected):
    assert _join_or(items) == expected
    assert _join_or(items) == qa_eval._join_or(items)


# --------------------------------------------------------------------------- #
# MixtralQAParser: reference second family (single user turn, leading space).
# --------------------------------------------------------------------------- #
class _RecordingTokenizer(FakeTokenizer):
    """FakeTokenizer that records the messages passed to apply_chat_template."""

    def __init__(self):
        self.seen_messages = None

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        self.seen_messages = messages
        return super().apply_chat_template(messages, tokenize=tokenize, **kwargs)


def test_mixtral_parser_is_qaparser_subclass():
    assert isinstance(MixtralQAParser(), QAParser)


def test_mixtral_default_leading_space_is_true():
    """Mistral tokenizers encode ' A', so the answer-token form is space-prefixed."""
    parser = MixtralQAParser()
    assert parser.leading_space is True
    ids = parser.resolve_answer_token_ids(FakeTokenizer(), ["A", "B"])
    assert ids == [ord(" "), ord(" ")]  # FakeTokenizer.encode -> [ord(text[0])]


def test_mixtral_format_prompt_uses_single_user_turn_no_system_role():
    """The instruction is folded into one user message (no system role)."""
    sample = SAMPLES["mmlu_int"]
    tok = _RecordingTokenizer()
    MixtralQAParser().format_prompt(sample, tok, _letters(sample))
    assert tok.seen_messages is not None
    assert len(tok.seen_messages) == 1
    assert tok.seen_messages[0]["role"] == "user"
    assert all(m["role"] != "system" for m in tok.seen_messages)


def test_mixtral_format_prompt_includes_question_options_and_letters():
    sample = SAMPLES["mmlu_int"]
    prompt = MixtralQAParser().format_prompt(sample, FakeTokenizer(), _letters(sample))
    assert "Q1?" in prompt
    assert "A. a" in prompt
    assert "A, B, C, or D" in prompt  # _join_or over the four letters
    assert "Answer:" in prompt


def test_mixtral_prompt_differs_from_qwen_two_message_form():
    """Mixtral (one user turn) and Qwen (system + user) produce different prompts."""
    sample = SAMPLES["mmlu_int"]
    letters = _letters(sample)
    mixtral = MixtralQAParser().format_prompt(sample, FakeTokenizer(), letters)
    qwen = Qwen15QAParser().format_prompt(sample, FakeTokenizer(), letters)
    assert mixtral != qwen


# --------------------------------------------------------------------------- #
# §3.7 housekeeping: abstract base, explicit concrete params.
# --------------------------------------------------------------------------- #
def test_abstract_base_cannot_be_instantiated():
    with pytest.raises(TypeError):
        QAParser()


def test_concrete_format_prompt_rejects_unknown_kwarg():
    sample = SAMPLES["mmlu_int"]
    with pytest.raises(TypeError):
        Qwen15QAParser().format_prompt(sample, FakeTokenizer(), _letters(sample), bogus=1)


def test_concrete_resolve_answer_token_ids_rejects_unknown_kwarg():
    with pytest.raises(TypeError):
        Qwen15QAParser().resolve_answer_token_ids(FakeTokenizer(), ["A", "B"], bogus=1)
