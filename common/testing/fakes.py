"""Fake model zoo for CPU tests and the golden harness.

Stands in for Qwen/Qwen1.5-MoE, its tokenizer, and datasets so pipeline stages
run end-to-end on CPU with no downloads. Kept tiny (see size constants below).

Do not import experiment config here -- this module is shared by both pipeline
test trees.

Determinism: FakeTokenizer maps text to a fixed id sequence (no RNG);
FakeCausalLM derives hidden states from a fixed embedding of those ids and
logits from a fixed linear head over the pooled per-layer routed states -- the
whole forward is RNG-free. Captured co-activation matrices, MLP inputs, and
top-k selections stay independent of forward-call order (a single pass produces
the same artifacts as the old per-stage separate passes), and the logits are a
deterministic function of the routed path, so the stage-4 accuracy gate actually
depends on which experts are fused instead of being a constant.
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


class FakeMixtralExpert(nn.Module):
    """Fake Mixtral-family expert (``MixtralBlockSparseTop2MLP`` layout).

    Same SiLU gate/up/down math as :class:`FakeExpert`, but with Mixtral's
    projection names -- ``w1`` (gate), ``w3`` (up), ``w2`` (down) -- so the
    ``MixtralExpertAdapter`` contract can be exercised without a real Mixtral.
    """

    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(H, INTER, bias=False)  # gate
        self.w2 = nn.Linear(INTER, H, bias=False)  # down
        self.w3 = nn.Linear(H, INTER, bias=False)  # up

    def forward(self, x):
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))


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
    """Fake MoE instead of Qwen.

    Routes tokens through their top-k experts and adds the gated shared expert,
    mirroring the real Qwen1.5-MoE block (and the fused ``FusionLayer`` that
    replaces it in stage 4, which likewise returns a tensor of the input shape).
    The forward is a deterministic, RNG-free function of its input, so swapping a
    layer's mlp for a fused one actually moves the model output -- which is what
    makes the stage-4 accuracy gate content-ful rather than a tautology (the old
    forward was the identity ``return x`` and ignored the experts entirely).
    """

    def __init__(self):
        super().__init__()
        self.gate = FakeRouterGate()
        self.experts = nn.ModuleList([FakeExpert() for _ in range(N_EXPERTS)])
        self.shared_expert = nn.Linear(H, H, bias=False)
        self.shared_expert_gate = nn.Linear(H, 1, bias=False)

    def forward(self, x):
        shape = x.shape
        tokens = x.reshape(-1, H)
        weights, selected = torch.topk(self.gate(tokens).softmax(dim=-1), TOP_K, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        routed = torch.zeros_like(tokens)
        for slot in range(TOP_K):
            for expert_id, expert in enumerate(self.experts):
                mask = selected[:, slot] == expert_id
                if mask.any():
                    routed[mask] += weights[mask, slot].unsqueeze(-1) * expert(tokens[mask])
        shared = torch.sigmoid(self.shared_expert_gate(tokens)) * self.shared_expert(tokens)
        return (routed + shared).reshape(shape)


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
        # Fixed embedding makes hidden states a deterministic function of input
        # ids; the fixed head makes logits a deterministic function of the pooled
        # routed states. Both are constructed after ``_FakeInner`` so the
        # expert/gate/shared weights (which feed the golden partition and
        # reconstruction) keep their original RNG draws unchanged.
        self.embed = nn.Embedding(VOCAB, H)
        self.lm_head = nn.Linear(H, VOCAB, bias=False)
        self.dtype = torch.float32
        self.device = torch.device("cpu")

    def to(self, *args, **kwargs):
        return self

    def forward(self, input_ids=None, attention_mask=None, *args, **kwargs):
        ids = input_ids if input_ids.dim() > 1 else input_ids.unsqueeze(0)
        ids = ids.long().to(self.embed.weight.device)
        hidden = self.embed(ids % VOCAB)
        # Every layer reads the same fixed-embedding hidden (no residual
        # chaining), so per-layer mlp inputs / co-activations stay call- and
        # layer-order-independent and byte-identical to the old identity forward
        # that the capture seam and golden partition/reconstruction are pinned
        # to. Logits come from the pooled per-layer routed outputs (no raw
        # embedding residual, which would otherwise dominate and pin the coarse
        # answer-token argmax) through a fixed head, so accuracy depends strongly
        # on every layer's experts: swapping in a fused layer (stage 4) moves the
        # answer-token argmax and therefore the measured accuracy.
        pooled = torch.stack([layer.mlp(hidden) for layer in self.model.layers]).sum(dim=0)
        return types.SimpleNamespace(logits=self.lm_head(pooled))

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
    """Fake dataset instead of MMLU.

    The answer labels are deliberately unbalanced. FakeTokenizer collapses every
    prompt to the same ids (it only sees the shared system-prompt prefix, not the
    question), so the model emits one constant prediction ``P`` for all samples;
    with a balanced label set accuracy would be pinned at ``1/n_choices`` for any
    ``P`` -- a tautology that hides whether stage 4 assembles and restores layers
    correctly. Distinct per-label counts (A:0, B:1, C:3, D:4 over 8 samples) make
    accuracy ``= count(label == P) / n`` a strictly monotone function of ``P``,
    and ``P`` depends on the routed experts, so fusing (or failing to restore) a
    layer moves the measured accuracy.
    """

    _LABELS = [1, 2, 2, 2, 3, 3, 3, 3]

    def __init__(self, n=8):
        self._samples = [
            {
                "question": f"q{i}",
                "choices": ["a", "b", "c", "d"],
                "answer": self._LABELS[i % len(self._LABELS)],
            }
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
