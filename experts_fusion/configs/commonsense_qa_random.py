"""E1 ablation cell: CommonsenseQA x random grouping (lower bound on CommonsenseQA).

Run with ``python -m experts_fusion.run --config commonsense_qa_random``.
"""
from experts_fusion.configs._base import *  # noqa: F401,F403 -- shared ablation surface
from experts_fusion.configs._base import (  # noqa: F401 -- re-export surface + dataset specs
    __all__,
    COMMONSENSE_QA_DATASETS,
)
from common.names import PairingMethod

DATA_DIR = "./data/ef_commonsense_qa_random"
TRAIN_DATASET_NAME = COMMONSENSE_QA_DATASETS["train"]
VALIDATION_DATASET_NAME = COMMONSENSE_QA_DATASETS["validation"]
TEST_DATASET_NAME = COMMONSENSE_QA_DATASETS["test"]
TEST_SIZE = COMMONSENSE_QA_DATASETS["test_size"]
PAIRING_METHOD = PairingMethod.RANDOM
