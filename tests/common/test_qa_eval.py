import pytest
import torch

from common import qa_eval


@pytest.fixture(autouse=True)
def _clear_cache():
    qa_eval._DATA_CACHE.clear()
    yield
    qa_eval._DATA_CACHE.clear()


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
    assert qa_eval._join_or(items) == expected


def test_cache_key_is_order_independent():
    k1 = qa_eval._cache_key({"path": "p", "split": "test"}, 10)
    k2 = qa_eval._cache_key({"split": "test", "path": "p"}, 10)
    assert k1 == k2
    assert k1 != qa_eval._cache_key({"path": "p", "split": "test"}, 11)


def test_extract_question_variants():
    assert qa_eval._extract_question({"question": "q"}) == "q"
    assert qa_eval._extract_question({"sentence": "s"}) == "s"
    assert qa_eval._extract_question({"question_stem": "qs"}) == "qs"
    with pytest.raises(KeyError):
        qa_eval._extract_question({"nope": 1})


def test_extract_choices_dict_form():
    sample = {"choices": {"label": ["A", "B"], "text": ["x", "y"]}}
    labels, texts = qa_eval._extract_choices(sample)
    assert labels == ["A", "B"]
    assert texts == ["x", "y"]


def test_extract_choices_list_form_generates_letters():
    sample = {"choices": ["x", "y", "z"]}
    labels, texts = qa_eval._extract_choices(sample)
    assert labels == ["A", "B", "C"]
    assert texts == ["x", "y", "z"]


def test_extract_choices_winogrande_options():
    sample = {"option1": "first", "option2": "second"}
    labels, texts = qa_eval._extract_choices(sample)
    assert labels == ["A", "B"]
    assert texts == ["first", "second"]


def test_extract_choices_unknown_raises():
    with pytest.raises(KeyError):
        qa_eval._extract_choices({"foo": "bar"})


def test_extract_answer_answer_key():
    assert qa_eval._extract_answer({"answerKey": "C"}, ["A", "B", "C"]) == "C"


def test_extract_answer_integer_index():
    assert qa_eval._extract_answer({"answer": 2}, ["A", "B", "C", "D"]) == "C"


def test_extract_answer_already_letter():
    assert qa_eval._extract_answer({"answer": "B"}, ["A", "B", "C"]) == "B"


def test_extract_answer_winogrande_numeric_string():
    assert qa_eval._extract_answer({"answer": "1"}, ["A", "B"]) == "A"
    assert qa_eval._extract_answer({"answer": "2"}, ["A", "B"]) == "B"


class _FakeBatch:
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def to(self, device):
        return self


class FakeTokenizer:
    def __init__(self):
        self.padding_side = "right"
        self.truncation_side = "right"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return " ".join(m["content"] for m in messages)

    def encode(self, text, add_special_tokens=False):
        return [ord(text[0])]  # letters map to their ASCII code (< vocab below)

    def __call__(self, prompts, return_tensors="pt", padding=True,
                 truncation=True, max_length=512):
        batch = len(prompts)
        ids = torch.zeros(batch, 4, dtype=torch.long)
        mask = torch.ones(batch, 4, dtype=torch.long)
        return _FakeBatch(ids, mask)


class _FakeOutput:
    def __init__(self, logits):
        self.logits = logits


class FakeModel:
    """Always confidently predicts answer letter 'A'."""

    VOCAB = 128

    def __init__(self):
        self.device = torch.device("cpu")
        self.training = False

    def eval(self):
        self.training = False
        return self

    def train(self):
        self.training = True
        return self

    def __call__(self, input_ids, attention_mask, output_hidden_states=False,
                 output_attentions=False, use_cache=False):
        batch, seq = input_ids.shape
        logits = torch.full((batch, seq, self.VOCAB), -10.0)
        logits[:, -1, ord("A")] = 10.0
        return _FakeOutput(logits)


class FakeDataset:
    def __init__(self, samples):
        self._samples = samples

    def __len__(self):
        return len(self._samples)

    def select(self, indices):
        return FakeDataset([self._samples[i] for i in indices])

    def __getitem__(self, i):
        return self._samples[i]

    def __iter__(self):
        return iter(self._samples)


def _make_samples(n=6):
    return [
        {
            "question": f"Question {i}?",
            "choices": {"label": ["A", "B", "C", "D"],
                        "text": ["a", "b", "c", "d"]},
            "answerKey": "A" if i % 2 == 0 else "B",
        }
        for i in range(n)
    ]


@pytest.fixture
def patch_dataset(monkeypatch):
    samples = _make_samples(6)
    calls = {"count": 0}

    def fake_load_dataset(**kwargs):
        calls["count"] += 1
        return FakeDataset(samples)

    monkeypatch.setattr(qa_eval, "load_dataset", fake_load_dataset)
    return calls


def test_format_prompt_includes_question_and_options():
    sample = _make_samples(1)[0]
    prompt = qa_eval._format_prompt(sample, FakeTokenizer(), ["A", "B", "C", "D"])
    assert "Question 0?" in prompt
    assert "A. a" in prompt


def test_resolve_answer_token_ids():
    ids = qa_eval._resolve_answer_token_ids(FakeTokenizer(), ["A", "B"])
    assert ids == [ord("A"), ord("B")]


def test_prepare_builds_expected_structure(patch_dataset):
    cache = qa_eval._prepare(
        {"path": "p", "split": "test"}, n=4, tokenizer=FakeTokenizer(), seed=42
    )
    assert len(cache["prompts"]) == 4
    assert len(cache["ground_truths"]) == 4
    assert cache["answer_letters"] == ["A", "B", "C", "D"]
    assert cache["letter_to_idx"]["A"] == 0


def test_evaluate_qa_returns_fraction_and_caches(patch_dataset):
    model = FakeModel()
    tokenizer = FakeTokenizer()
    kwargs = {"path": "p", "split": "test"}

    acc = qa_eval.evaluate_qa(model, tokenizer, kwargs, n=6, seed=42, batch_size=2)
    assert acc == pytest.approx(0.5)
    assert patch_dataset["count"] == 1

    qa_eval.evaluate_qa(model, tokenizer, kwargs, n=6, seed=42, batch_size=2)
    assert patch_dataset["count"] == 1


def test_evaluate_qa_sets_eval_and_left_padding(patch_dataset):
    model = FakeModel()
    tokenizer = FakeTokenizer()
    qa_eval.evaluate_qa(model, tokenizer, {"path": "p", "split": "test"}, n=4, seed=1)
    assert tokenizer.padding_side == "left"
    assert tokenizer.truncation_side == "left"
    assert model.training is False


def test_evaluate_qa_restores_training_mode(patch_dataset):
    model = FakeModel()
    model.train()
    assert model.training is True
    qa_eval.evaluate_qa(model, FakeTokenizer(), {"path": "p", "split": "test"}, n=4)
    assert model.training is True