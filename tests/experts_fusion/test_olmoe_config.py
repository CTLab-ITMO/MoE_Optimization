"""Config-selection tests for the E3 second-model cell (OLMoE-1B-7B on ARC-Challenge).

``olmoe_arc_challenge`` retargets the whole pipeline at a second MoE family
(``allenai/OLMoE-1B-7B-0924-Instruct``) to rebut "fusion only works on Qwen1.5-MoE".
Unlike the E1/E2 cells it does NOT ``import *`` from ``_base`` (which hardcodes the Qwen
model, sizes, and Qwen shared-expert adapter): the model, its routing constants
(EXPERTS/TOP_K/LAYERS), and all four DI adapters are set locally for OLMoE.

These tests are the step's CPU-only validation (the GPU run is a later step). They pin
(a) the scoped/distinct DATA_DIR, (b) the OLMoE routing constants, (c) that every DI
adapter is the correct existing class with the load-bearing non-default constructor args
(``normalize_topk=False``, ``leading_space=True``, ``NoSharedExpertAdapter``), (d) the ARC
dataset triple on the labeled test split, (e) that the config surface matches
``default.__all__`` and survives the ``experts_fusion.config`` dispatch, (f) the mirrored
E-series FILTER decision #2, and (g) that this cell genuinely differs from the Qwen family
(different model, different expert count, no shared expert).
"""
import importlib

import pytest
import torch

from common.expert_adapter import (
    NoSharedExpertAdapter,
    Qwen15ExpertAdapter,
    SharedExpertAdapter,
)
from common.fusion import FUSION_REGISTRY, ZipItFusion
from common.grouping import GROUPING_REGISTRY
from common.names import PairingMethod
from common.qa_parser import Qwen15QAParser
from common.router import Qwen15RouterAdapter

CONFIG_NAME = "olmoe_arc_challenge"
OLMOE_MODEL = "allenai/OLMoE-1B-7B-0924-Instruct"
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
# Per-cell scoped output dir, distinct from every other config.
# --------------------------------------------------------------------------- #
def test_data_dir_is_scoped_and_named():
    cfg = _load()
    assert cfg.DATA_DIR == f"./data/ef_{CONFIG_NAME}"
    assert cfg.DATA_DIR.startswith("./data/")


def test_data_dir_distinct_from_e1_and_arc_cells():
    """OLMoE must not collide with the six E1 cells or the E2 ARC cell."""
    from tests.experts_fusion.test_ablation_configs import CELLS

    other_dirs = {importlib.import_module(f"experts_fusion.configs.{n}").DATA_DIR for n, _, _ in CELLS}
    other_dirs.add(importlib.import_module("experts_fusion.configs.arc_challenge_activation_correlation").DATA_DIR)
    assert _load().DATA_DIR not in other_dirs


# --------------------------------------------------------------------------- #
# Second-model routing constants (the point of E3).
# --------------------------------------------------------------------------- #
def test_targets_olmoe_model_family():
    cfg = _load()
    assert cfg.MODEL_NAME == OLMOE_MODEL
    assert cfg.MODEL_DTYPE == torch.bfloat16


def test_olmoe_routing_constants():
    """OLMoE: 64 experts, top-8, 16 all-MoE layers -- distinct from Qwen1.5-MoE's 60/4/24."""
    cfg = _load()
    assert cfg.EXPERTS == 64
    assert cfg.TOP_K == 8
    assert cfg.LAYERS == list(range(16))
    # Even expert count so pair-grouping (k=2) partitions cleanly.
    assert cfg.EXPERTS % 2 == 0


def test_differs_from_qwen_default_family():
    """E3's whole point: a genuinely different model from the Qwen1.5-MoE default."""
    from experts_fusion.configs import default

    cfg = _load()
    assert cfg.MODEL_NAME != default.MODEL_NAME
    assert cfg.EXPERTS != default.EXPERTS
    assert cfg.TOP_K != default.TOP_K


# --------------------------------------------------------------------------- #
# DI adapters: existing classes, load-bearing non-default constructor args.
# --------------------------------------------------------------------------- #
def test_expert_adapter_is_qwen15_reused_for_olmoe():
    """OLMoE's gate_proj/up_proj/down_proj experts are covered by Qwen15ExpertAdapter."""
    cfg = _load()
    assert isinstance(cfg.EXPERT_ADAPTER, Qwen15ExpertAdapter)


def test_router_adapter_does_not_normalize_topk():
    """OLMoE sets norm_topk_prob=false, so top-k softmax mass is NOT renormalized."""
    cfg = _load()
    assert isinstance(cfg.ROUTER_ADAPTER, Qwen15RouterAdapter)
    assert cfg.ROUTER_ADAPTER.normalize_topk is False


def test_shared_expert_adapter_is_the_no_shared_fallback():
    """OLMoE has no shared expert -> the fallback adapter, extracting nothing."""
    cfg = _load()
    assert isinstance(cfg.SHARED_EXPERT_ADAPTER, NoSharedExpertAdapter)
    assert isinstance(cfg.SHARED_EXPERT_ADAPTER, SharedExpertAdapter)
    # Contract: extract yields None and build yields a zeros-contributing module.
    assert cfg.SHARED_EXPERT_ADAPTER.extract(object()) is None
    runtime = cfg.SHARED_EXPERT_ADAPTER.build(None)
    assert runtime.has_shared is False


def test_qa_parser_uses_leading_space_for_byte_level_bpe():
    """OLMoE's GPT-NeoX BPE tokenizes ' A' distinctly, so the answer token is space-prefixed."""
    cfg = _load()
    assert isinstance(cfg.QA_PARSER, Qwen15QAParser)
    assert cfg.QA_PARSER.leading_space is True


# --------------------------------------------------------------------------- #
# ARC-Challenge dataset triple on the labeled test split.
# --------------------------------------------------------------------------- #
def test_binds_arc_challenge_dataset():
    cfg = _load()
    for spec in (cfg.TRAIN_DATASET_NAME, cfg.VALIDATION_DATASET_NAME, cfg.TEST_DATASET_NAME):
        assert spec["path"] == ARC_PATH
        assert spec["name"] == ARC_CONFIG
        assert "split" in spec
    assert cfg.DATASET_TEXT_FIELD == "question"


def test_evaluates_on_labeled_test_split():
    cfg = _load()
    assert cfg.TRAIN_DATASET_NAME["split"] == "train"
    assert cfg.VALIDATION_DATASET_NAME["split"] == "validation"
    assert cfg.TEST_DATASET_NAME["split"] == "test"
    assert cfg.TEST_SIZE == 1_172  # full ARC-Challenge test split


# --------------------------------------------------------------------------- #
# Config surface + FILTER decision + dispatch.
# --------------------------------------------------------------------------- #
def test_exposes_the_full_config_surface():
    from experts_fusion.configs import default

    cfg = _load()
    assert set(cfg.__all__) == set(default.__all__)
    missing = [field for field in cfg.__all__ if not hasattr(cfg, field)]
    assert not missing, f"{CONFIG_NAME} is missing config fields {missing}"


def test_pairing_method_is_the_primary_strategy():
    cfg = _load()
    assert cfg.PAIRING_METHOD == PairingMethod.ACTIVATION_CORRELATION
    assert cfg.PAIRING_METHOD in GROUPING_REGISTRY


def test_filter_matches_e_series_decision_two():
    """Decision #2 (mirrored from _base): real methods only, no Best_*/Distillation."""
    cfg = _load()
    assert isinstance(cfg.FILTER_FUSIONS, list) and cfg.FILTER_FUSIONS
    assert "Distillation" not in cfg.FILTER_FUSIONS
    assert "LS_Distillation" not in cfg.FILTER_FUSIONS
    assert ZipItFusion.name in cfg.FILTER_FUSIONS
    assert all(method in FUSION_REGISTRY for method in cfg.FILTER_FUSIONS)
    assert cfg.FILTER_METRICS == []
    assert cfg.INSPECT_FUSION_METHOD in cfg.FILTER_FUSIONS


def test_config_dispatch_reexports_olmoe_cell(select_config):
    config = select_config(CONFIG_NAME)
    leaf = _load()
    assert config._CONFIG_NAME == CONFIG_NAME
    assert config.MODEL_NAME == leaf.MODEL_NAME
    assert config.EXPERTS == leaf.EXPERTS
    assert config.TOP_K == leaf.TOP_K
    assert config.LAYERS == leaf.LAYERS
    # DATA_DIR is dropped from the snapshot; the model constants round-trip.
    snapshot = config.config_snapshot_dict()
    assert snapshot["MODEL_NAME"] == OLMOE_MODEL
    assert snapshot["EXPERTS"] == 64
    assert snapshot["TOP_K"] == 8
    assert snapshot["TEST_DATASET_NAME"]["name"] == ARC_CONFIG
    assert "Distillation" not in snapshot["FILTER_FUSIONS"]
