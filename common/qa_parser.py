"""QA prompt builder and answer-token resolver, behind an ABC.

:class:`QAParser` abstracts two things so adding a new model family means
writing a subclass, not editing the eval loop:

* ``format_prompt`` builds the chat prompt via ``tokenizer.apply_chat_template``;
* ``resolve_answer_token_ids`` returns the first token id for each answer letter.

:class:`Qwen15QAParser` implements the Qwen1.5 chat format. The ``leading_space``
flag controls whether a letter is encoded as ``"A"`` (default) or ``" A"``.
:class:`MixtralQAParser` is the reference second family: a single ``user`` turn
(no ``system`` role, which the Mistral template lacks) and ``leading_space=True``.

Library only -- never imports an experiment ``config``.
"""

from abc import ABC, abstractmethod

from common import dataset_schema


def _join_or(items: list) -> str:
    """Join a letter list into a readable phrase.

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

    Concrete parsers take explicit named parameters.
    """

    @abstractmethod
    def format_prompt(self, sample: dict, tokenizer, letters: list) -> str:
        """Render the chat prompt string for ``sample``."""
        raise NotImplementedError()

    @abstractmethod
    def resolve_answer_token_ids(self, tokenizer, letters: list) -> list:
        """First-token id of each candidate answer ``letter``."""
        raise NotImplementedError()


class Qwen15QAParser(QAParser):
    """Qwen1.5 chat-format :class:`QAParser`.

    ``leading_space=False`` (default) encodes each answer letter as ``"A"``;
    ``True`` encodes it as ``" A"`` for tokenizers that prefix word boundaries.
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


class MixtralQAParser(Qwen15QAParser):
    """QA parser for Mixtral / Mistral-family chat models.

    Two differences from :class:`Qwen15QAParser`:

    * the Mistral ``[INST] ... [/INST]`` template historically carries no
      ``system`` role, so the answer-format instruction is folded into a single
      ``user`` turn instead of being sent as a separate system message (a system
      message would raise on tokenizers whose template rejects that role);
    * ``leading_space`` defaults to ``True`` because the Mistral tokenizer encodes
      an answer letter as a space-prefixed token (``" A"``), so the first-token id
      is resolved from that form.

    ``resolve_answer_token_ids`` is inherited unchanged (it already honours
    ``leading_space``).
    """

    def __init__(self, leading_space: bool = True):
        super().__init__(leading_space=leading_space)

    def format_prompt(self, sample: dict, tokenizer, letters: list) -> str:
        labels, texts = dataset_schema.resolve_choices(sample)
        options_text = ""
        for label, text in zip(labels, texts):
            options_text += f"{label}. {text}\n"

        user_message = (
            "You are a precise evaluation system. "
            "You will be given a question and multiple-choice options. "
            f"Answer with ONLY the letter corresponding to the correct option ({_join_or(letters)}). "
            "Do not write the full answer, do not write explanations. Output only one character.\n\n"
            f"Question: {dataset_schema.resolve_question(sample)}\n"
            f"Options:\n{options_text}\nAnswer:"
        )
        messages = [{"role": "user", "content": user_message}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
