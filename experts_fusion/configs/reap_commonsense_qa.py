"""REAP baseline: CommonsenseQA x router-weighted expert-activation pruning.

Same REAP pruning path as ``reap_mmlu`` (see its module docstring), on the
CommonsenseQA axis. Accuracy is measured on the labeled validation split because
CommonsenseQA ships its test split without gold labels.

Run with ``python -m experts_fusion.run --config reap_commonsense_qa``.
"""
from experts_fusion.configs._base import *  # noqa: F401,F403 -- shared ablation surface
from experts_fusion.configs._base import (  # noqa: F401 -- re-export surface + dataset specs
    __all__,
    COMMONSENSE_QA_DATASETS,
)
from common.fusion import PruningFusion
from common.names import PairingMethod

DATA_DIR = "./data/ef_reap_commonsense_qa"
TRAIN_DATASET_NAME = COMMONSENSE_QA_DATASETS["train"]
VALIDATION_DATASET_NAME = COMMONSENSE_QA_DATASETS["validation"]
TEST_DATASET_NAME = COMMONSENSE_QA_DATASETS["test"]
TEST_SIZE = COMMONSENSE_QA_DATASETS["test_size"]

PAIRING_METHOD = PairingMethod.REAP_SALIENCY
FILTER_FUSIONS = [PruningFusion.name]
INSPECT_FUSION_METHOD = PruningFusion.name
