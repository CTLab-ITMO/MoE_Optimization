"""Shared fake model zoo and artifact builders for tests.

No ``config`` dependency, so both legacy and new pipeline tests share these fakes.
"""
from common.testing.builders import Builders, expert_mlp
from common.testing.fakes import (
    H,
    INTER,
    N_EXPERTS,
    N_LAYERS,
    TOP_K,
    VOCAB,
    FakeCausalLM,
    FakeExpert,
    FakeFusionGate,
    FakeLayer,
    FakeMoEMLP,
    FakeQADataset,
    FakeRouterGate,
    FakeTokenizer,
    make_fake_datasets,
    make_fake_transformers,
)

__all__ = [
    "Builders",
    "expert_mlp",
    "H",
    "INTER",
    "N_EXPERTS",
    "N_LAYERS",
    "TOP_K",
    "VOCAB",
    "FakeCausalLM",
    "FakeExpert",
    "FakeFusionGate",
    "FakeLayer",
    "FakeMoEMLP",
    "FakeQADataset",
    "FakeRouterGate",
    "FakeTokenizer",
    "make_fake_datasets",
    "make_fake_transformers",
]
