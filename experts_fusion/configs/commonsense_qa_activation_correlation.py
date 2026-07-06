"""E1 ablation cell: CommonsenseQA x activation_correlation grouping.

Overrides the dataset triple to CommonsenseQA (accuracy on the labeled validation
split; the test split ships without gold labels). Grouping stays at the base default
(activation_correlation). Run with
``python -m experts_fusion.run --config commonsense_qa_activation_correlation``.
"""
from experts_fusion.configs._base import *  # noqa: F401,F403 -- shared ablation surface
from experts_fusion.configs._base import (  # noqa: F401 -- re-export surface + dataset specs
    __all__,
    COMMONSENSE_QA_DATASETS,
)

DATA_DIR = "./data/ef_commonsense_qa_activation_correlation"
TRAIN_DATASET_NAME = COMMONSENSE_QA_DATASETS["train"]
VALIDATION_DATASET_NAME = COMMONSENSE_QA_DATASETS["validation"]
TEST_DATASET_NAME = COMMONSENSE_QA_DATASETS["test"]
TEST_SIZE = COMMONSENSE_QA_DATASETS["test_size"]
