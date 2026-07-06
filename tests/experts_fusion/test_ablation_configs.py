"""Config-selection tests for the E1 grouping-ablation grid.

Six configs live under ``experts_fusion/configs/``:
``{mmlu, commonsense_qa} x {activation_correlation, random, output_similarity}``.
Each scopes its outputs under its own ``DATA_DIR`` and pins one ``PAIRING_METHOD``;
they share every other knob via ``_base`` so the ablation is apples-to-apples.

These tests pin (a) the grid is complete and its cells are distinct, (b) each
cell's per-cell overrides (DATA_DIR / dataset / pairing), (c) the shared E1
decision-#2 FILTER surface, and (d) that ``experts_fusion.config`` dispatches on
the ``EXPERTS_FUSION_CONFIG`` env var -- exactly what ``run.py --config`` sets.
"""
import importlib

import pytest

from common.fusion import FUSION_REGISTRY
from common.grouping import GROUPING_REGISTRY
from common.names import PairingMethod

# Expected shape of each task axis: dataset HF path and the split used for the
# labeled accuracy set (CommonsenseQA's test split has no gold labels).
DATASET_AXES = {
    "mmlu": {"path": "cais/mmlu", "test_split": "test", "test_size": 15_000},
    "commonsense_qa": {"path": "tau/commonsense_qa", "test_split": "validation", "test_size": 1_221},
}
PAIRINGS = (
    PairingMethod.ACTIVATION_CORRELATION,
    PairingMethod.RANDOM,
    PairingMethod.OUTPUT_SIMILARITY,
)
# The six ablation cells: (config_name, dataset_key, pairing).
CELLS = [
    (f"{dataset}_{pairing}", dataset, pairing)
    for dataset in DATASET_AXES
    for pairing in PAIRINGS
]


def _load(name):
    return importlib.import_module(f"experts_fusion.configs.{name}")


@pytest.fixture
def select_config(monkeypatch):
    """Select a config via the env var and reload ``experts_fusion.config``.

    Mirrors what ``run.py --config`` does at process start. Restores the default
    config on teardown so later tests still see ``experts_fusion.config`` as default.
    """
    import experts_fusion.config as config

    def _select(name):
        monkeypatch.setenv("EXPERTS_FUSION_CONFIG", name)
        return importlib.reload(config)

    yield _select
    monkeypatch.delenv("EXPERTS_FUSION_CONFIG", raising=False)
    importlib.reload(config)


def test_grid_is_the_full_two_by_three_ablation():
    assert len(CELLS) == 6
    assert {name for name, _, _ in CELLS} == {
        "mmlu_activation_correlation",
        "mmlu_random",
        "mmlu_output_similarity",
        "commonsense_qa_activation_correlation",
        "commonsense_qa_random",
        "commonsense_qa_output_similarity",
    }


def test_every_cell_data_dir_is_distinct():
    dirs = [_load(name).DATA_DIR for name, _, _ in CELLS]
    assert len(set(dirs)) == len(dirs), dirs


@pytest.mark.parametrize("name,dataset,pairing", CELLS)
def test_cell_data_dir_is_scoped_and_named(name, dataset, pairing):
    cfg = _load(name)
    assert cfg.DATA_DIR == f"./data/ef_{name}"
    # Every cell writes under ./data/ (run from the repo root).
    assert cfg.DATA_DIR.startswith("./data/")


@pytest.mark.parametrize("name,dataset,pairing", CELLS)
def test_cell_pins_its_pairing_method(name, dataset, pairing):
    cfg = _load(name)
    assert cfg.PAIRING_METHOD == pairing
    # The strategy must be resolvable by stage 2.
    assert cfg.PAIRING_METHOD in GROUPING_REGISTRY


@pytest.mark.parametrize("name,dataset,pairing", CELLS)
def test_cell_binds_the_right_dataset(name, dataset, pairing):
    cfg = _load(name)
    axis = DATASET_AXES[dataset]
    for spec in (cfg.TRAIN_DATASET_NAME, cfg.VALIDATION_DATASET_NAME, cfg.TEST_DATASET_NAME):
        assert spec["path"] == axis["path"]
        assert "split" in spec
    # Accuracy is measured on the labeled split (CommonsenseQA test is unlabeled).
    assert cfg.TEST_DATASET_NAME["split"] == axis["test_split"]
    assert cfg.TEST_SIZE == axis["test_size"]


def test_commonsense_qa_evaluates_on_labeled_validation_split():
    """CommonsenseQA test labels are hidden; accuracy must run on validation."""
    for name, dataset, _ in CELLS:
        if dataset != "commonsense_qa":
            continue
        cfg = _load(name)
        assert cfg.TEST_DATASET_NAME["split"] == "validation"
        assert cfg.TRAIN_DATASET_NAME["split"] == "train"


@pytest.mark.parametrize("name,dataset,pairing", CELLS)
def test_cell_exposes_the_full_config_surface(name, dataset, pairing):
    from experts_fusion.configs import default

    cfg = _load(name)
    assert set(cfg.__all__) == set(default.__all__)
    missing = [field for field in cfg.__all__ if not hasattr(cfg, field)]
    assert not missing, f"{name} is missing config fields {missing}"


@pytest.mark.parametrize("name,dataset,pairing", CELLS)
def test_cell_filter_matches_e1_decision_two(name, dataset, pairing):
    """Decision #2: compare on real methods, drop Best_* and Distillation."""
    cfg = _load(name)
    assert isinstance(cfg.FILTER_FUSIONS, list) and cfg.FILTER_FUSIONS
    # No Distillation (also drops its LS_ wrapper); strong methods stay.
    assert "Distillation" not in cfg.FILTER_FUSIONS
    assert "LS_Distillation" not in cfg.FILTER_FUSIONS
    assert "ZipIt" in cfg.FILTER_FUSIONS
    # Every filtered method must be registered (stages assert this too).
    assert all(method in FUSION_REGISTRY for method in cfg.FILTER_FUSIONS)
    # Empty metric filter -> stage 3 produces no Best_<metric> pseudo-methods.
    assert cfg.FILTER_METRICS == []
    # Stage-4 diagnostics need INSPECT_FUSION_METHOD to survive the filter.
    assert cfg.INSPECT_FUSION_METHOD in cfg.FILTER_FUSIONS


@pytest.mark.parametrize("name,dataset,pairing", CELLS)
def test_config_dispatch_reexports_selected_cell(name, dataset, pairing, select_config):
    """experts_fusion.config re-exports the cell chosen via EXPERTS_FUSION_CONFIG."""
    config = select_config(name)
    leaf = _load(name)
    assert config._CONFIG_NAME == name
    assert config.DATA_DIR == leaf.DATA_DIR
    assert config.PAIRING_METHOD == leaf.PAIRING_METHOD
    assert config.TEST_DATASET_NAME == leaf.TEST_DATASET_NAME
    assert config.TEST_SIZE == leaf.TEST_SIZE
    # DATA_DIR is dropped from the snapshot; the rest round-trips.
    snapshot = config.config_snapshot_dict()
    assert snapshot["PAIRING_METHOD"] == leaf.PAIRING_METHOD
    assert "Distillation" not in snapshot["FILTER_FUSIONS"]


def test_config_dispatch_round_trips_back_to_default(monkeypatch):
    """Clearing the env var and reloading returns config to the default cell.

    Self-contained (no shared fixture): selecting a non-default cell must not
    leave experts_fusion.config stuck away from default for the rest of the suite.
    """
    import experts_fusion.config as config

    monkeypatch.setenv("EXPERTS_FUSION_CONFIG", "commonsense_qa_random")
    importlib.reload(config)
    assert config._CONFIG_NAME == "commonsense_qa_random"

    monkeypatch.delenv("EXPERTS_FUSION_CONFIG", raising=False)
    importlib.reload(config)
    assert config._CONFIG_NAME == "default"
    assert config.DATA_DIR == "./data/data_experts_fusion"
