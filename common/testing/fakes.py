"""Importable fake model zoo for CPU tests and the golden harness.

A self-contained, ``config``-free copy of the fakes that used to live only in
``tests/experiment/conftest.py``. These stand in for ``Qwen/Qwen1.5-MoE`` and
its tokenizer/datasets so the pipeline stages can run end-to-end on CPU with no
downloads. Kept deliberately tiny (see the size constants below) so a full
stage 1..9 chain runs in seconds.

This module must not import any experiment ``config`` -- it is library code
shared by both the legacy and the new pipeline test trees.
"""
import types

import torch
from torch import nn

# Fake-model dimensions. Small on purpose: enough experts to form pairs, enough
# layers to exercise per-layer logic, tiny hidden/intermediate widths for speed.
H = 4
INTER = 6
N_EXPERTS = 4
N_LAYERS = 2
TOP_K = 2
VOCAB = 128


class _BatchEncoding(dict):
    def to(self, *args, **kwargs):
        return self

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


class FakeExpert(nn.Module):
    """Fake expert."""

    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(H, INTER, bias=False)
        self.up_proj = nn.Linear(H, INTER, bias=False)
        self.down_proj = nn.Linear(INTER, H, bias=False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class FakeRouterGate(nn.Module):
    """Fake router gate."""

    def __init__(self):
        super().__init__()
        self.num_experts = N_EXPERTS
        self.proj = nn.Linear(H, N_EXPERTS, bias=False)

    def forward(self, x):
        return self.proj(x)


class FakeFusionGate(nn.Module):
    """Fake gate for fusion, returns a 3-tuple."""

    def __init__(self):
        super().__init__()
        self.num_experts = N_EXPERTS
        self.proj = nn.Linear(H, N_EXPERTS, bias=False)

    def forward(self, x):
        logits = self.proj(x)
        weights, selected = torch.topk(logits.softmax(dim=-1), TOP_K, dim=-1)
        return logits, weights, selected


class FakeMoEMLP(nn.Module):
    """Fake MoE instead of Qwen."""

    def __init__(self):
        super().__init__()
        self.gate = FakeRouterGate()
        self.experts = nn.ModuleList([FakeExpert() for _ in range(N_EXPERTS)])
        self.shared_expert = nn.Linear(H, H, bias=False)
        self.shared_expert_gate = nn.Linear(H, 1, bias=False)

    def forward(self, x):
        self.gate(x.reshape(-1, H))
        return x


class FakeLayer(nn.Module):
    """Fake layer with only MoE."""

    def __init__(self):
        super().__init__()
        self.mlp = FakeMoEMLP()


class _FakeInner(nn.Module):
    def __init__(self, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([FakeLayer() for _ in range(n_layers)])


class FakeCausalLM(nn.Module):
    """Fake Qwen."""

    def __init__(self, n_layers=N_LAYERS + 1):
        super().__init__()
        self.model = _FakeInner(n_layers)
        self.dtype = torch.float32
        self.device = torch.device("cpu")

    def to(self, *args, **kwargs):
        return self

    def forward(self, input_ids=None, attention_mask=None, *args, **kwargs):
        batch = input_ids.shape[0]
        seq = input_ids.shape[1] if input_ids.dim() > 1 else input_ids.shape[0]
        hidden = torch.randn(batch, seq, H)
        for layer in self.model.layers:
            hidden = layer.mlp(hidden)
        return types.SimpleNamespace(logits=torch.randn(batch, seq, VOCAB))

    def __call__(self, input_ids=None, **kwargs):
        return self.forward(input_ids=input_ids, **kwargs)

    def generate(self, input_ids=None, **kwargs):
        self.forward(input_ids=input_ids)
        return input_ids


class FakeTokenizer:
    def __call__(self, text, return_tensors=None, **kwargs):
        batch = len(text) if isinstance(text, list) else 1
        return _BatchEncoding(
            input_ids=torch.randint(0, 10, (batch, 4)),
            attention_mask=torch.ones(batch, 4, dtype=torch.long),
        )

    def decode(self, ids, **kwargs):
        return "decoded"

    def encode(self, text, add_special_tokens=False):
        return [ord(text[0])]

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        if tokenize:
            return _BatchEncoding(
                input_ids=torch.randint(0, 10, (1, 4)),
                attention_mask=torch.ones(1, 4, dtype=torch.long),
            )
        return " ".join(m["content"] for m in messages)


def make_fake_transformers():
    mod = types.ModuleType("transformers")
    mod.AutoModelForCausalLM = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: FakeCausalLM()
    )
    mod.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: FakeTokenizer()
    )
    return mod


def make_fake_datasets():
    mod = types.ModuleType("datasets")
    mod.load_dataset = lambda *a, **k: {"question": [f"question {i}" for i in range(8)]}
    return mod


class FakeQADataset:
    """Fake dataset instead of MMLU"""

    def __init__(self, n=8):
        self._samples = [
            {"question": f"q{i}", "choices": ["a", "b", "c", "d"], "answer": i % 4}
            for i in range(n)
        ]

    def __len__(self):
        return len(self._samples)

    def select(self, indices):
        ds = FakeQADataset.__new__(FakeQADataset)
        ds._samples = [self._samples[i] for i in indices]
        return ds

    def __getitem__(self, i):
        return self._samples[i]

    def __iter__(self):
        return iter(self._samples)
