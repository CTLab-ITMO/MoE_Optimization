"""HC-SMoE baseline: CommonsenseQA x hierarchical-clustering expert merging.

Same HC-SMoE merging path as ``hcsmoe_mmlu`` (see its module docstring), on the
CommonsenseQA axis. Accuracy is measured on the labeled validation split because
CommonsenseQA ships its test split without gold labels.

Run with ``python -m experts_fusion.run --config hcsmoe_commonsense_qa``.
"""
from experts_fusion.configs._base import *  # noqa: F401,F403 -- shared ablation surface
from experts_fusion.configs._base import (  # noqa: F401 -- re-export surface + dataset specs
    __all__,
    COMMONSENSE_QA_DATASETS,
)
from common.baselines.hcsmoe import FrequencyWeightedAverage
from common.names import PairingMethod

DATA_DIR = "./data/ef_hcsmoe_commonsense_qa"
TRAIN_DATASET_NAME = COMMONSENSE_QA_DATASETS["train"]
VALIDATION_DATASET_NAME = COMMONSENSE_QA_DATASETS["validation"]
TEST_DATASET_NAME = COMMONSENSE_QA_DATASETS["test"]
TEST_SIZE = COMMONSENSE_QA_DATASETS["test_size"]

PAIRING_METHOD = PairingMethod.HCSMOE_CLUSTERING
FILTER_FUSIONS = [FrequencyWeightedAverage.name]
INSPECT_FUSION_METHOD = FrequencyWeightedAverage.name
