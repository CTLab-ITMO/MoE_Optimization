"""Config-selection tests for the REAP / HC-SMoE baseline grid (step 7).

Six configs live under ``experts_fusion/configs/``:
``{reap, hcsmoe} x {mmlu, commonsense_qa, arc_challenge}``. Each ports an external
MoE-compression method into our pipeline so the paper can report baseline numbers
directly comparable to our fusion:

* ``reap``   -> ``PAIRING_METHOD="reap_saliency"``,  ``FILTER_FUSIONS=["Pruning"]``
* ``hcsmoe`` -> ``PAIRING_METHOD="hcsmoe_clustering"``, ``FILTER_FUSIONS=["FrequencyWeightedAverage"]``

Each scopes its outputs under its own ``DATA_DIR`` (``./data/ef_reap_*`` /
``./data/ef_hcsmoe_*``) and reuses the E1 ``_base`` surface so the comparison stays
apples-to-apples. These tests pin (a) the grid is complete and its cells distinct,
(b) each cell's per-cell overrides (DATA_DIR / dataset / pairing), (c) the single
baseline fusion method survives ``FILTER_FUSIONS`` and matches
``INSPECT_FUSION_METHOD`` (the base default ``ZipIt`` would fail stage 4), and
(d) ``experts_fusion.config`` dispatches on ``EXPERTS_FUSION_CONFIG``. They are
purely additive -- the six E1 cells and the E2 ARC cell are pinned elsewhere and
left untouched.
"""
import importlib

import pytest

from common.fusion import FUSION_REGISTRY
from common.grouping import GROUPING_REGISTRY
from common.names import PairingMethod

# Baseline methods: grouping strategy + the single fusion method its groups use.
METHODS = {
    "reap": {"pairing": PairingMethod.REAP_SALIENCY, "fusion": "Pruning"},
    "hcsmoe": {"pairing": PairingMethod.HCSMOE_CLUSTERING, "fusion": "FrequencyWeightedAverage"},
}
# Dataset axes: HF path, the split used for the labeled accuracy set, and (ARC only)
# the multi-config name that must be passed through to load_dataset.
DATASET_AXES = {
    "mmlu": {"path": "cais/mmlu", "test_split": "test", "test_size": 15_000, "name": "all"},
    "commonsense_qa": {"path": "tau/commonsense_qa", "test_split": "validation", "test_size": 1_221, "name": None},
    "arc_challenge": {"path": "allenai/ai2_arc", "test_split": "test", "test_size": 1_172, "name": "ARC-Challenge"},
}
# The six baseline cells: (config_name, method_key, dataset_key). Method-first names
# make DATA_DIR ``./data/ef_reap_*`` / ``./data/ef_hcsmoe_*`` fall out of the shared
# ``./data/ef_{name}`` convention.
CELLS = [
    (f"{method}_{dataset}", method, dataset)
    for method in METHODS
    for dataset in DATASET_AXES
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


def test_grid_is_the_full_two_by_three_baseline():
    assert len(CELLS) == 6
    assert {name for name, _, _ in CELLS} == {
        "reap_mmlu",
        "reap_commonsense_qa",
        "reap_arc_challenge",
        "hcsmoe_mmlu",
        "hcsmoe_commonsense_qa",
        "hcsmoe_arc_challenge",
    }


def test_every_cell_data_dir_is_distinct():
    dirs = [_load(name).DATA_DIR for name, _, _ in CELLS]
    assert len(set(dirs)) == len(dirs), dirs


@pytest.mark.parametrize("name,method,dataset", CELLS)
def test_cell_data_dir_is_scoped_and_named(name, method, dataset):
    cfg = _load(name)
    assert cfg.DATA_DIR == f"./data/ef_{name}"
    # Method-first DATA_DIR prefix per STEPS step 7 (./data/ef_reap_* / ef_hcsmoe_*).
    assert cfg.DATA_DIR.startswith(f"./data/ef_{method}_")


def test_baseline_dirs_distinct_from_e1_and_e2_grids():
    """The baselines must not collide with any E1 ablation cell or the E2 ARC cell."""
    from tests.experts_fusion.test_ablation_configs import CELLS as E1_CELLS

    prior = {_load(n).DATA_DIR for n, _, _ in E1_CELLS}
    prior.add(_load("arc_challenge_activation_correlation").DATA_DIR)
    baseline = {_load(n).DATA_DIR for n, _, _ in CELLS}
    assert baseline.isdisjoint(prior), baseline & prior


@pytest.mark.parametrize("name,method,dataset", CELLS)
def test_cell_pins_its_pairing_method(name, method, dataset):
    cfg = _load(name)
    assert cfg.PAIRING_METHOD == METHODS[method]["pairing"]
    # The strategy must be resolvable by stage 2.
    assert cfg.PAIRING_METHOD in GROUPING_REGISTRY


@pytest.mark.parametrize("name,method,dataset", CELLS)
def test_cell_binds_the_right_dataset(name, method, dataset):
    cfg = _load(name)
    axis = DATASET_AXES[dataset]
    for spec in (cfg.TRAIN_DATASET_NAME, cfg.VALIDATION_DATASET_NAME, cfg.TEST_DATASET_NAME):
        assert spec["path"] == axis["path"]
        assert "split" in spec
        if axis["name"] is not None:
            # Multi-config datasets (MMLU "all", ARC "ARC-Challenge") pass name through.
            assert spec["name"] == axis["name"]
    # Accuracy is measured on the labeled split (CommonsenseQA test is unlabeled).
    assert cfg.TEST_DATASET_NAME["split"] == axis["test_split"]
    assert cfg.TEST_SIZE == axis["test_size"]


def test_commonsense_qa_evaluates_on_labeled_validation_split():
    """CommonsenseQA test labels are hidden; accuracy must run on validation."""
    for name, _, dataset in CELLS:
        if dataset != "commonsense_qa":
            continue
        cfg = _load(name)
        assert cfg.TEST_DATASET_NAME["split"] == "validation"
        assert cfg.TRAIN_DATASET_NAME["split"] == "train"


@pytest.mark.parametrize("name,method,dataset", CELLS)
def test_cell_exposes_the_full_config_surface(name, method, dataset):
    from experts_fusion.configs import default

    cfg = _load(name)
    assert set(cfg.__all__) == set(default.__all__)
    missing = [field for field in cfg.__all__ if not hasattr(cfg, field)]
    assert not missing, f"{name} is missing config fields {missing}"


@pytest.mark.parametrize("name,method,dataset", CELLS)
def test_cell_narrows_to_its_single_baseline_method(name, method, dataset):
    """Each baseline runs exactly one fusion method, and stage-4 inspection uses it."""
    cfg = _load(name)
    fusion = METHODS[method]["fusion"]
    assert cfg.FILTER_FUSIONS == [fusion]
    # The one method must be registered (stages assert this too).
    assert fusion in FUSION_REGISTRY
    # Best_* pseudo-methods stay off (inherited from _base).
    assert cfg.FILTER_METRICS == []
    # Stage-4 diagnostics need INSPECT_FUSION_METHOD to survive the filter; the base
    # default "ZipIt" is not produced here and would fail stage 4 at start.
    assert cfg.INSPECT_FUSION_METHOD == fusion
    assert cfg.INSPECT_FUSION_METHOD in cfg.FILTER_FUSIONS


@pytest.mark.parametrize("name,method,dataset", CELLS)
def test_config_dispatch_reexports_selected_cell(name, method, dataset, select_config):
    """experts_fusion.config re-exports the cell chosen via EXPERTS_FUSION_CONFIG."""
    config = select_config(name)
    leaf = _load(name)
    assert config._CONFIG_NAME == name
    assert config.DATA_DIR == leaf.DATA_DIR
    assert config.PAIRING_METHOD == leaf.PAIRING_METHOD
    assert config.FILTER_FUSIONS == leaf.FILTER_FUSIONS
    assert config.INSPECT_FUSION_METHOD == leaf.INSPECT_FUSION_METHOD
    assert config.TEST_DATASET_NAME == leaf.TEST_DATASET_NAME
    assert config.TEST_SIZE == leaf.TEST_SIZE
    # DATA_DIR is dropped from the snapshot; the rest round-trips.
    snapshot = config.config_snapshot_dict()
    assert snapshot["PAIRING_METHOD"] == leaf.PAIRING_METHOD
    assert snapshot["FILTER_FUSIONS"] == [METHODS[method]["fusion"]]
    assert snapshot["INSPECT_FUSION_METHOD"] == METHODS[method]["fusion"]


def test_arc_baselines_carry_the_multiconfig_name():
    """ARC is a multi-config dataset; both ARC baselines must pass ARC-Challenge through."""
    for name in ("reap_arc_challenge", "hcsmoe_arc_challenge"):
        cfg = _load(name)
        for spec in (cfg.TRAIN_DATASET_NAME, cfg.VALIDATION_DATASET_NAME, cfg.TEST_DATASET_NAME):
            assert spec["name"] == "ARC-Challenge"
