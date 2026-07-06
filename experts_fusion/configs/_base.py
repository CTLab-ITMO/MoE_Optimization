"""Shared base for the E1 grouping-ablation configs.

The E1 ablation sweeps the grid ``{mmlu, commonsense_qa}`` x
``{activation_correlation, random, output_similarity}`` -- one config per cell under
``experts_fusion/configs/``. Every cell shares this module's knobs (model, adapters,
sizes, ``FILTER_*``) so the comparison is apples-to-apples; each leaf overrides only
what its cell differs in: ``DATA_DIR``, the dataset triple/``TEST_SIZE``, and
``PAIRING_METHOD``. A leaf pulls the shared surface with
``from experts_fusion.configs._base import *`` and re-exports ``__all__``.

This module's defaults are the ``mmlu`` x ``activation_correlation`` cell, so that
leaf overrides only ``DATA_DIR``. It is a helper, not a run target: the leading
underscore keeps it out of the ``--config`` surface (pass a leaf name instead).

Note vs ``default.py``: ``default`` is the general pipeline default and runs every
fusion method (``FILTER_* = None``). These ablation cells narrow the surface per
Костя's E1 decision #2 -- compare on the real stage-3 methods, dropping the
``Best_*`` pseudo-methods (``FILTER_METRICS = []`` -> none produced) and the
expensive ``Distillation`` method -- to keep every cell identical and save GPU hours.
"""
import torch

from common.expert_adapter import Qwen15ExpertAdapter
from common.fusion import FUSION_REGISTRY, ZipItFusion
from common.names import PairingMethod
from common.qa_parser import Qwen15QAParser
from common.router import Qwen15RouterAdapter

# --------------------------------------------------------------------------- #
# Dataset triples for the two task axes. Helpers (not config knobs), so they are
# NOT in ``__all__`` and never reach a leaf via ``import *``; a leaf that needs the
# commonsense_qa axis imports ``COMMONSENSE_QA_DATASETS`` explicitly. Centralising
# the specs here keeps paths/splits single-sourced across the three cells per axis.
# --------------------------------------------------------------------------- #
MMLU_DATASETS = {
    "train": {"path": "cais/mmlu", "name": "all", "split": "auxiliary_train"},
    "validation": {"path": "cais/mmlu", "name": "all", "split": "validation"},
    "test": {"path": "cais/mmlu", "name": "all", "split": "test"},
    "test_size": 15_000,
}
COMMONSENSE_QA_DATASETS = {
    "train": {"path": "tau/commonsense_qa", "split": "train"},
    "validation": {"path": "tau/commonsense_qa", "split": "validation"},
    # CommonsenseQA ships its test split without gold labels, so accuracy is
    # evaluated on the labeled validation split (see E2/E1 dataset notes).
    "test": {"path": "tau/commonsense_qa", "split": "validation"},
    "test_size": 1_221,  # full CommonsenseQA validation set
}

# --------------------------------------------------------------------------- #
# Per-cell axes -- leaf configs override these. Base defaults to mmlu x
# activation_correlation.
# --------------------------------------------------------------------------- #
DATA_DIR = "./data/ef_mmlu_activation_correlation"

TRAIN_DATASET_NAME = MMLU_DATASETS["train"]
VALIDATION_DATASET_NAME = MMLU_DATASETS["validation"]
TEST_DATASET_NAME = MMLU_DATASETS["test"]
TEST_SIZE = MMLU_DATASETS["test_size"]

# Expert-grouping strategy; one of: activation_correlation, random, output_similarity.
PAIRING_METHOD = PairingMethod.ACTIVATION_CORRELATION

# --------------------------------------------------------------------------- #
# Shared across every ablation cell.
# --------------------------------------------------------------------------- #
MODEL_NAME = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
MODEL_DTYPE = torch.bfloat16
EXPERTS = 60

# Input text field read by stages 1 and 4; "question" holds for both MMLU and
# CommonsenseQA.
DATASET_TEXT_FIELD = "question"

TRAIN_SIZE = 1_024       # train set for reconstruction and layer-order selection
VALIDATION_SIZE = 1_024  # held-out set for reconstruction validation
LAYERS = list(range(24))
TOP_K = 4

REPLACE_LAYERS_SEQUENTIAL = False
REPLACE_BEST_LAYERS_SEQUENTIAL = True
REPLACE_LAYERS_SMART_SEQUENTIAL = False

# False: load only fused experts (production mode); True: keep originals for A/B.
USE_DEFAULT_EXPERTS = False

# Stage 3: score a group only when it has more than this many validation activations.
MIN_PAIR_VALIDATION_SAMPLES = 10

# Method whose fused experts stage 4 inspects (recon + generation). Must survive the
# FILTER_FUSIONS filter below -- ZipIt does (only Distillation is dropped).
INSPECT_FUSION_METHOD = ZipItFusion.name

SEED = 42

# Model-family DI adapters; the Qwen1.5 instances are dataset-agnostic (the QA
# parser auto-detects the sample schema), so both task axes share them.
ROUTER_ADAPTER = Qwen15RouterAdapter()
EXPERT_ADAPTER = Qwen15ExpertAdapter()
QA_PARSER = Qwen15QAParser()

# E1 decision #2: compare grouping strategies on the real stage-3 methods only.
# FILTER_METRICS = [] suppresses the Best_<metric> pseudo-methods; excluding
# "Distillation" by substring also drops its LS_Distillation wrapper (the two
# costliest methods). Shared by all six cells so the ablation stays apples-to-apples.
FILTER_FUSIONS = [name for name in FUSION_REGISTRY.names() if "Distillation" not in name]
FILTER_METRICS: list = []

# The config surface re-exported by experts_fusion.config (leaves re-export this).
__all__ = [
    "DATA_DIR",
    "MODEL_NAME",
    "MODEL_DTYPE",
    "EXPERTS",
    "TRAIN_DATASET_NAME",
    "VALIDATION_DATASET_NAME",
    "TEST_DATASET_NAME",
    "DATASET_TEXT_FIELD",
    "TRAIN_SIZE",
    "VALIDATION_SIZE",
    "TEST_SIZE",
    "LAYERS",
    "TOP_K",
    "REPLACE_LAYERS_SEQUENTIAL",
    "REPLACE_BEST_LAYERS_SEQUENTIAL",
    "REPLACE_LAYERS_SMART_SEQUENTIAL",
    "USE_DEFAULT_EXPERTS",
    "MIN_PAIR_VALIDATION_SAMPLES",
    "INSPECT_FUSION_METHOD",
    "SEED",
    "ROUTER_ADAPTER",
    "EXPERT_ADAPTER",
    "QA_PARSER",
    "PAIRING_METHOD",
    "FILTER_FUSIONS",
    "FILTER_METRICS",
]
