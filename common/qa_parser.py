"""Model-family QA prompt / answer-token parser (§3.2).

The legacy ``common.qa_eval`` builds the evaluation prompt and resolves the
candidate answer-letter token ids inline, hard-wired to the Qwen1.5 chat format:

* ``_format_prompt`` -- formats the options block, a fixed system instruction
  (with an ``A or B`` / ``A, B, or C`` enumeration of the valid letters) and the
  question, then renders them through ``tokenizer.apply_chat_template(...)``;
* ``_resolve_answer_token_ids`` -- encodes each *bare* letter (``"A"``, no
  leading space) and takes the first token id.

This module lifts those two responsibilities behind a :class:`QAParser` ABC so a
new model family is a new subclass rather than an edit to the eval loop. The
in-tree :class:`Qwen15QAParser` reproduces both inline functions EXACTLY by
default. The single observed degree of freedom is the §3.2 "leading space"
subtlety -- whether the answer letter is encoded as ``"A"`` or ``" A"`` -- exposed
as a constructor flag whose DEFAULT (``leading_space=False``) reproduces the
current behaviour.

Per §3.7 the variadic ``**kwargs`` lives only on the *abstract* signatures (for
forward-compatible contract freezing); the concrete parser takes explicit, named
parameters so a typo is caught rather than silently swallowed.

Library code only -- this module never imports an experiment ``config``.
"""

from abc import ABC, abstractmethod

from common import dataset_schema


def _join_or(items: list) -> str:
    """Human-readable enumeration of letters (== ``qa_eval._join_or``).

    ``[]`` -> ``""``; ``["A"]`` -> ``"A"``; ``["A", "B"]`` -> ``"A or B"``;
    ``["A", "B", "C"]`` -> ``"A, B, or C"``.
    """
    if len(items) <= 1:
        return "".join(items)
    if len(items) == 2:
        return f"{items[0]} or {items[1]}"
    return ", ".join(items[:-1]) + f", or {items[-1]}"


class QAParser(ABC):
    """Build the evaluation prompt and the candidate answer-token ids.

    ``**kwargs`` is the forward-compatible interface slot only (§3.7); concrete
    parsers take explicit named parameters.
    """

    @abstractmethod
    def format_prompt(self, sample: dict, tokenizer, letters: list, **kwargs) -> str:
        """Render the chat prompt string for ``sample``."""
        raise NotImplementedError()

    @abstractmethod
    def resolve_answer_token_ids(self, tokenizer, letters: list, **kwargs) -> list:
        """First-token id of each candidate answer ``letter``."""
        raise NotImplementedError()


class Qwen15QAParser(QAParser):
    """Qwen1.5 chat-format parser; reproduces the inline ``qa_eval`` logic.

    ``leading_space`` controls how each answer letter is encoded when resolving
    its token id (§3.2): the DEFAULT ``False`` encodes the bare letter ``"A"``
    (reproducing ``qa_eval._resolve_answer_token_ids`` exactly); ``True`` encodes
    a leading-space variant ``" A"`` for tokenizers that prefix word boundaries.
    """

    def __init__(self, leading_space: bool = False):
        self.leading_space = leading_space

    def format_prompt(self, sample: dict, tokenizer, letters: list) -> str:
        labels, texts = dataset_schema.resolve_choices(sample)
        options_text = ""
        for label, text in zip(labels, texts):
            options_text += f"{label}. {text}\n"

        system_message = (
            "You are a precise evaluation system. "
            "You will be given a question and multiple-choice options. "
            f"Your task is to answer with ONLY the letter corresponding to the correct option ({_join_or(letters)}). "
            "Do not write the full answer, do not write explanations. Output only one character."
        )
        user_message = (
            f"Question: {dataset_schema.resolve_question(sample)}\n"
            f"Options:\n{options_text}\nAnswer:"
        )
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def resolve_answer_token_ids(self, tokenizer, letters: list) -> list:
        ids = []
        for letter in letters:
            token = f" {letter}" if self.leading_space else letter
            toks = tokenizer.encode(token, add_special_tokens=False)
            ids.append(toks[0])
        return ids
