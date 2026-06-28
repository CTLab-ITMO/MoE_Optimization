"""Fixtures for the new ``experts_fusion`` pipeline tests.

Mirrors ``tests/experiment/conftest.py`` but targets the new module, which is
launched from the repository root: stages import ``experts_fusion.config`` and
``common.*`` as packages (repo root is already on ``sys.path`` via the root
``tests/conftest.py``), so there is no per-module ``sys.path`` manipulation and
no bare ``config`` import. The fake model zoo is imported from
``common.testing`` instead of being redefined here.
"""
import runpy
import sys
from pathlib import Path

import pytest
import torch

from common.testing import (
    N_EXPERTS,
    TOP_K,
    FakeQADataset,
    make_fake_datasets,
    make_fake_transformers,
)
from common.testing.fakes import N_LAYERS

ROOT = Path(__file__).resolve().parents[2]
NEW_MODULE_DIR = ROOT / "experts_fusion"

CPU = torch.device("cpu")


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
    from common.testing import Builders

    return Builders(data_dir, layers, N_EXPERTS)


@pytest.fixture
def run_script(monkeypatch, tmp_path, data_dir, layers):
    def _run(name, *, config_overrides=None, with_model=False, patch_qa_eval=False):
        import experts_fusion.config as cfg

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
            monkeypatch.setitem(sys.modules, "transformers", make_fake_transformers())
            monkeypatch.setitem(sys.modules, "datasets", make_fake_datasets())

        if patch_qa_eval:
            from common import qa_eval
            monkeypatch.setattr(qa_eval, "load_dataset", lambda *a, **k: FakeQADataset())
            qa_eval._DATA_CACHE.clear()

        workdir = tmp_path / "work"
        workdir.mkdir(exist_ok=True)
        monkeypatch.chdir(workdir)

        runpy.run_path(str(NEW_MODULE_DIR / name), run_name="__main__")

    return _run
