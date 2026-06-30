"""Fake model zoo for CPU tests and the golden harness.

Stands in for Qwen/Qwen1.5-MoE, its tokenizer, and datasets so pipeline stages
run end-to-end on CPU with no downloads. Kept tiny (see size constants below).

Do not import experiment config here -- this module is shared by both pipeline
test trees.

Determinism: FakeTokenizer maps text to a fixed id sequence (no RNG);
FakeCausalLM derives hidden states from a fixed embedding of those ids rather
than calling torch.randn. As a result, captured co-activation matrices, MLP
inputs, and top-k selections are independent of forward-call order, so a single
pass produces the same artifacts as the old per-stage separate passes.
Logits stay random; evaluate_qa reseeds before consuming them.
"""
import types

import torch
from torch import nn

# Model dimensions, kept small: enough experts for pairing, enough layers for
# per-layer logic, tiny hidden/intermediate widths for speed.
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
        # Fixed embedding makes hidden states a deterministic function of input ids.
        # Constructed last so expert/gate/shared weights keep their RNG draws.
        self.embed = nn.Embedding(VOCAB, H)
        self.dtype = torch.float32
        self.device = torch.device("cpu")

    def to(self, *args, **kwargs):
        return self

    def forward(self, input_ids=None, attention_mask=None, *args, **kwargs):
        ids = input_ids if input_ids.dim() > 1 else input_ids.unsqueeze(0)
        ids = ids.long().to(self.embed.weight.device)
        batch, seq = ids.shape[0], ids.shape[1]
        # Fixed-embedding hidden states are call-order-independent, so a single
        # forward pass captures the same mlp_inputs and co-activations as
        # separate per-stage passes would.
        hidden = self.embed(ids % VOCAB)
        for layer in self.model.layers:
            hidden = layer.mlp(hidden)
        # logits stay random: evaluate_qa reseeds before consuming them, so
        # accuracy is reproducible without coupling it to the hidden path.
        return types.SimpleNamespace(logits=torch.randn(batch, seq, VOCAB))

    def __call__(self, input_ids=None, **kwargs):
        return self.forward(input_ids=input_ids, **kwargs)

    def generate(self, input_ids=None, **kwargs):
        self.forward(input_ids=input_ids)
        return input_ids


# Fixed fake sequence length; ids are derived from the text, not sampled.
_FAKE_SEQ_LEN = 4


def _deterministic_ids(text: str, length: int = _FAKE_SEQ_LEN) -> list[int]:
    """Map a string to a fixed-length id sequence (no RNG).

    Distinct strings get distinct ids; mapping is stable across runs and
    processes, so single-pass capture produces consistent artifacts.
    """
    ids = [(ord(ch) + position) % VOCAB for position, ch in enumerate(text[:length])]
    while len(ids) < length:
        # Pad deterministically from the text length so short strings still vary.
        ids.append((len(text) * 31 + len(ids) * 7) % VOCAB)
    return ids


class FakeTokenizer:
    def __call__(self, text, return_tensors=None, **kwargs):
        texts = text if isinstance(text, list) else [text]
        input_ids = torch.tensor(
            [_deterministic_ids(str(t)) for t in texts], dtype=torch.long
        )
        attention_mask = torch.ones(
            input_ids.shape[0], input_ids.shape[1], dtype=torch.long
        )
        return _BatchEncoding(input_ids=input_ids, attention_mask=attention_mask)

    def decode(self, ids, **kwargs):
        return "decoded"

    def encode(self, text, add_special_tokens=False):
        return [ord(text[0])]

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        joined = " ".join(m["content"] for m in messages)
        if tokenize:
            input_ids = torch.tensor([_deterministic_ids(joined)], dtype=torch.long)
            return _BatchEncoding(
                input_ids=input_ids,
                attention_mask=torch.ones(1, input_ids.shape[1], dtype=torch.long),
            )
        return joined


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
