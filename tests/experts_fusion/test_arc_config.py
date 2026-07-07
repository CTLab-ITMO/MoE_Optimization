"""Config-selection tests for the E2 ARC-Challenge cell.

``arc_challenge_activation_correlation`` adds a third eval dataset on top of the
E1 grid (``allenai/ai2_arc``, config ``ARC-Challenge``). It reuses the whole
``_base`` ablation surface and overrides only the dataset triple, its scoped
``DATA_DIR`` and ``TEST_SIZE``.

These tests pin (a) the per-cell overrides (DATA_DIR / dataset / labeled test
split), (b) that the shared knobs survive ``import *`` (text field, pairing,
parser, FILTER decision #2, full config surface), (c) that ``experts_fusion.config``
dispatches on ``EXPERTS_FUSION_CONFIG`` the way ``run.py --config`` does, and
(d) that ``common.dataset_schema`` actually resolves ARC-shaped samples --
including ARC's numeric labels and 3/5-choice items, resolved per sample.
"""
import importlib

import pytest

from common import dataset_schema
from common.dataset_schema import detect_schema
from common.fusion import FUSION_REGISTRY
from common.grouping import GROUPING_REGISTRY
from common.names import PairingMethod
from common.qa_parser import Qwen15QAParser

CONFIG_NAME = "arc_challenge_activation_correlation"
ARC_PATH = "allenai/ai2_arc"
ARC_CONFIG = "ARC-Challenge"


def _load(name=CONFIG_NAME):
    return importlib.import_module(f"experts_fusion.configs.{name}")


@pytest.fixture
def select_config(monkeypatch):
    """Select a config via the env var and reload ``experts_fusion.config``.

    Mirrors ``run.py --config`` at process start; restores the default config on
    teardown so later tests still see ``experts_fusion.config`` as default.
    """
    import experts_fusion.config as config

    def _select(name):
        monkeypatch.setenv("EXPERTS_FUSION_CONFIG", name)
        return importlib.reload(config)

    yield _select
    monkeypatch.delenv("EXPERTS_FUSION_CONFIG", raising=False)
    importlib.reload(config)


# --------------------------------------------------------------------------- #
# Per-cell overrides: scoped output dir + ARC dataset triple.
# --------------------------------------------------------------------------- #
def test_data_dir_is_scoped_and_named():
    cfg = _load()
    assert cfg.DATA_DIR == f"./data/ef_{CONFIG_NAME}"
    assert cfg.DATA_DIR.startswith("./data/")


def test_data_dir_distinct_from_e1_grid():
    """ARC must not collide with any of the six E1 ablation cells."""
    from tests.experts_fusion.test_ablation_configs import CELLS

    e1_dirs = {importlib.import_module(f"experts_fusion.configs.{n}").DATA_DIR for n, _, _ in CELLS}
    assert _load().DATA_DIR not in e1_dirs


def test_binds_arc_challenge_dataset():
    cfg = _load()
    for spec in (cfg.TRAIN_DATASET_NAME, cfg.VALIDATION_DATASET_NAME, cfg.TEST_DATASET_NAME):
        assert spec["path"] == ARC_PATH
        # allenai/ai2_arc is multi-config; the config name must be passed through.
        assert spec["name"] == ARC_CONFIG
        assert "split" in spec


def test_evaluates_on_labeled_test_split():
    """Every ARC split is labeled, so accuracy runs on the held-out test split."""
    cfg = _load()
    assert cfg.TRAIN_DATASET_NAME["split"] == "train"
    assert cfg.VALIDATION_DATASET_NAME["split"] == "validation"
    assert cfg.TEST_DATASET_NAME["split"] == "test"
    assert cfg.TEST_SIZE == 1_172  # full ARC-Challenge test split


# --------------------------------------------------------------------------- #
# Shared surface survives ``import *`` from ``_base``.
# --------------------------------------------------------------------------- #
def test_text_field_and_pairing_inherited_from_base():
    cfg = _load()
    # ARC's question text lives under "question", same as MMLU/CommonsenseQA.
    assert cfg.DATASET_TEXT_FIELD == "question"
    assert cfg.PAIRING_METHOD == PairingMethod.ACTIVATION_CORRELATION
    assert cfg.PAIRING_METHOD in GROUPING_REGISTRY


def test_qa_parser_is_qwen15_autodetecting():
    cfg = _load()
    assert isinstance(cfg.QA_PARSER, Qwen15QAParser)


def test_exposes_the_full_config_surface():
    from experts_fusion.configs import default

    cfg = _load()
    assert set(cfg.__all__) == set(default.__all__)
    missing = [field for field in cfg.__all__ if not hasattr(cfg, field)]
    assert not missing, f"{CONFIG_NAME} is missing config fields {missing}"


def test_filter_matches_e1_decision_two():
    """Decision #2 (inherited from _base): real methods only, no Best_*/Distillation."""
    cfg = _load()
    assert isinstance(cfg.FILTER_FUSIONS, list) and cfg.FILTER_FUSIONS
    assert "Distillation" not in cfg.FILTER_FUSIONS
    assert "LS_Distillation" not in cfg.FILTER_FUSIONS
    assert "ZipIt" in cfg.FILTER_FUSIONS
    assert all(method in FUSION_REGISTRY for method in cfg.FILTER_FUSIONS)
    assert cfg.FILTER_METRICS == []
    assert cfg.INSPECT_FUSION_METHOD in cfg.FILTER_FUSIONS


# --------------------------------------------------------------------------- #
# Dispatch through experts_fusion.config (what run.py --config does).
# --------------------------------------------------------------------------- #
def test_config_dispatch_reexports_arc_cell(select_config):
    config = select_config(CONFIG_NAME)
    leaf = _load()
    assert config._CONFIG_NAME == CONFIG_NAME
    assert config.DATA_DIR == leaf.DATA_DIR
    assert config.TEST_DATASET_NAME == leaf.TEST_DATASET_NAME
    assert config.TEST_SIZE == leaf.TEST_SIZE
    # DATA_DIR is dropped from the snapshot; the rest round-trips.
    snapshot = config.config_snapshot_dict()
    assert snapshot["TEST_DATASET_NAME"]["name"] == ARC_CONFIG
    assert snapshot["DATASET_TEXT_FIELD"] == "question"
    assert "Distillation" not in snapshot["FILTER_FUSIONS"]


# --------------------------------------------------------------------------- #
# Schema coverage: the resolvers actually handle ARC-shaped samples.
# --------------------------------------------------------------------------- #
def test_schema_resolves_lettered_arc_sample():
    """Canonical ARC sample (4 lettered choices) resolves via the commonsense_qa schema."""
    sample = {
        "id": "Mercury_7175875",
        "question": "Which change of state releases energy?",
        "choices": {"label": ["A", "B", "C", "D"], "text": ["melting", "freezing", "boiling", "subliming"]},
        "answerKey": "B",
    }
    # ARC is key-identical to CommonsenseQA, so detect_schema routes it there.
    assert detect_schema(sample).name == "commonsense_qa"
    assert dataset_schema.resolve_question(sample) == sample["question"]
    labels, texts = dataset_schema.resolve_choices(sample)
    assert labels == ["A", "B", "C", "D"]
    assert texts == ["melting", "freezing", "boiling", "subliming"]
    assert dataset_schema.resolve_answer(sample, labels) == "B"


@pytest.mark.parametrize(
    "labels,key",
    [
        (["1", "2", "3", "4"], "3"),   # ARC's occasional numeric labels
        (["A", "B", "C"], "C"),        # 3-choice item
        (["A", "B", "C", "D", "E"], "E"),  # 5-choice item
    ],
)
def test_schema_resolves_heterogeneous_arc_samples(labels, key):
    """Numeric labels and 3/5-choice items resolve per sample (not assumed from A-D)."""
    sample = {
        "question": "Q?",
        "choices": {"label": labels, "text": [f"opt{l}" for l in labels]},
        "answerKey": key,
    }
    got_labels, _ = dataset_schema.resolve_choices(sample)
    assert got_labels == labels
    # answerKey is read straight from the sample -- no letter/number assumption.
    assert dataset_schema.resolve_answer(sample, got_labels) == key
