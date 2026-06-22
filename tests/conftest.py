import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = ROOT / "Qwen_MoE_experts_fusion_experiment"
TESTS_DIR = Path(__file__).resolve().parent

for _path in (str(ROOT), str(EXPERIMENT_DIR), str(TESTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import torch
import pytest

torch.cuda.is_available = lambda: False
torch.cuda.empty_cache = lambda: None
torch.cuda.synchronize = lambda *args, **kwargs: None

torch.set_num_threads(1)

CPU = torch.device("cpu")


@pytest.fixture
def cpu():
    return CPU


@pytest.fixture(autouse=True)
def _deterministic():
    torch.manual_seed(0)
    yield


@pytest.fixture
def make_advanced_mlp():
    """Factory make a small MLP with reproducible weights."""
    from common.mlp import AdvancedMLP

    def _factory(in_features=4, hidden=6, out_features=4, seed=0):
        torch.manual_seed(seed)
        return AdvancedMLP(in_features, hidden, out_features, device=CPU)

    return _factory


@pytest.fixture
def make_simple_mlp():
    """Factory make a small MLP with reproducible weights."""
    from common.mlp import SimpleMLP

    def _factory(in_features=3, hidden=4, out_features=4, seed=0):
        torch.manual_seed(seed)
        return SimpleMLP(in_features, hidden, out_features, device=CPU)

    return _factory


@pytest.fixture
def rand_tokens():
    def _factory(n=16, features=4, seed=1):
        torch.manual_seed(seed)
        return torch.randn(n, features, device=CPU)

    return _factory
