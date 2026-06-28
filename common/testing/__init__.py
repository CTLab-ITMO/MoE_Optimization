"""Importable fake model zoo and artifact builders for tests and golden runs.

Library code (no experiment ``config`` dependency) so both the legacy and the
new pipeline test trees -- and the golden harness -- share one source of fakes.
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
