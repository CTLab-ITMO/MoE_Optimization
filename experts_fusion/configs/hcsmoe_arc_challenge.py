"""HC-SMoE baseline: ARC-Challenge x hierarchical-clustering expert merging.

Same HC-SMoE merging path as ``hcsmoe_mmlu`` (see its module docstring), on the
ARC-Challenge axis (``allenai/ai2_arc``, config ``ARC-Challenge``). Every ARC split
is labeled, so accuracy runs on the held-out test split. The ARC dataset triple is
single-sourced from the E2 ``arc_challenge_activation_correlation`` cell.

Run with ``python -m experts_fusion.run --config hcsmoe_arc_challenge``.
"""
from experts_fusion.configs._base import *  # noqa: F401,F403 -- shared ablation surface
from experts_fusion.configs._base import __all__  # noqa: F401 -- re-export config surface
from experts_fusion.configs.arc_challenge_activation_correlation import ARC_CHALLENGE_DATASETS
from common.baselines.hcsmoe import FrequencyWeightedAverage
from common.names import PairingMethod

DATA_DIR = "./data/ef_hcsmoe_arc_challenge"
TRAIN_DATASET_NAME = ARC_CHALLENGE_DATASETS["train"]
VALIDATION_DATASET_NAME = ARC_CHALLENGE_DATASETS["validation"]
TEST_DATASET_NAME = ARC_CHALLENGE_DATASETS["test"]
TEST_SIZE = ARC_CHALLENGE_DATASETS["test_size"]

PAIRING_METHOD = PairingMethod.HCSMOE_CLUSTERING
FILTER_FUSIONS = [FrequencyWeightedAverage.name]
INSPECT_FUSION_METHOD = FrequencyWeightedAverage.name
