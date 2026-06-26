import os
import runpy
import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = ROOT / "Qwen_MoE_experts_fusion_experiment"

CPU = torch.device("cpu")
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
        self.device = CPU

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


def _make_fake_transformers():
    mod = types.ModuleType("transformers")
    mod.AutoModelForCausalLM = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: FakeCausalLM()
    )
    mod.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: FakeTokenizer()
    )
    return mod


def _make_fake_datasets():
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


def _expert_mlp(seed=0):
    from common.mlp import AdvancedMLP
    torch.manual_seed(seed)
    return AdvancedMLP(H, INTER, H, device=CPU)


class _Builders:
    """Generate fake ``.pt`` artifacts."""

    def __init__(self, data_dir, layers, n_experts):
        self.data_dir = data_dir
        self.layers = layers
        self.n_experts = n_experts

    def _path(self, *parts):
        path = os.path.join(self.data_dir, *parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def ensure_dir(self, *parts):
        os.makedirs(os.path.join(self.data_dir, *parts), exist_ok=True)

    def activation_matrices(self):
        for layer in self.layers:
            m = torch.rand(self.n_experts, self.n_experts)
            m = m + m.t()
            torch.save(m, self._path("activations", f"train_matrix_{layer}.pt"))
            torch.save(m.clone(), self._path("activations", f"validation_matrix_{layer}.pt"))

    def pairs(self):
        pairs = [(2 * i, 2 * i + 1) for i in range(self.n_experts // 2)]
        for layer in self.layers:
            torch.save(pairs, self._path("pairs", f"{layer}.pt"))
        return pairs

    def experts(self):
        for layer in self.layers:
            for i in range(self.n_experts):
                torch.save(_expert_mlp(i), self._path("experts", str(layer), f"expert_{i}.pt"))
            torch.save(FakeFusionGate(), self._path("experts", str(layer), "gate.pt"))
            torch.save(nn.Linear(H, H, bias=False),
                       self._path("experts", str(layer), "shared_expert.pt"))
            torch.save(nn.Linear(H, 1, bias=False),
                       self._path("experts", str(layer), "shared_expert_gate.pt"))

    def mlp_inputs(self, n=16):
        for layer in self.layers:
            torch.save([torch.randn(H) for _ in range(n)],
                       self._path("mlp_inputs", f"train_{layer}.pt"))
            torch.save([torch.randn(H) for _ in range(n)],
                       self._path("mlp_inputs", f"validation_{layer}.pt"))

    def pairs_activations(self, pairs, n=16):
        for layer in self.layers:
            torch.save([list(range(n)) for _ in pairs],
                       self._path("activations", f"pairs_train_{layer}.pt"))
            torch.save([list(range(n)) for _ in pairs],
                       self._path("activations", f"pairs_validation_{layer}.pt"))

    def fusion_experts(self, name, pairs):
        for layer in self.layers:
            for i in range(len(pairs)):
                torch.save(_expert_mlp(i),
                           self._path("fusion_experts", name, str(layer), f"{i}.pt"))


@pytest.fixture
def data_dir(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    return str(d)


@pytest.fixture
def layers():
    return list(range(N_LAYERS))


@pytest.fixture
def builders(data_dir, layers):
    return _Builders(data_dir, layers, N_EXPERTS)


@pytest.fixture
def run_script(monkeypatch, tmp_path, data_dir, layers):
    def _run(name, *, config_overrides=None, with_model=False, patch_qa_eval=False):
        import config as cfg

        overrides = {
            "DATA_DIR": data_dir,
            "LAYERS": layers,
            "EXPERTS": N_EXPERTS,
            "TOP_K": TOP_K,
            "device": CPU,
            "EVOLUTION_STEPS": 0,
            "PAIRING_METHOD": "activation_correlation",
            "TRAIN_SIZE": 4,
            "TEST_SIZE": 4,
            "VALIDATION_SIZE": 4,
            "FILTER_FUSIONS": ["Average"],
            "FILTER_METRICS": [],
            "REPLACE_LAYERS_SEQUANTIAL": False,
            "REPLACE_BEST_LAYERS_SEQUANTIAL": False,
            "REPLACE_LAYERS_SMART_SEQUANTIAL": False,
        }
        overrides.update(config_overrides or {})
        for key, value in overrides.items():
            monkeypatch.setattr(cfg, key, value, raising=False)

        import seaborn
        monkeypatch.setattr(seaborn, "kdeplot", lambda *a, **k: None)

        if with_model:
            monkeypatch.setitem(sys.modules, "transformers", _make_fake_transformers())
            monkeypatch.setitem(sys.modules, "datasets", _make_fake_datasets())

        if patch_qa_eval:
            from common import qa_eval
            monkeypatch.setattr(qa_eval, "load_dataset", lambda *a, **k: FakeQADataset())
            qa_eval._DATA_CACHE.clear()

        workdir = tmp_path / "work"
        workdir.mkdir(exist_ok=True)
        monkeypatch.chdir(workdir)

        runpy.run_path(str(EXPERIMENT_DIR / name), run_name="__main__")

    return _run
