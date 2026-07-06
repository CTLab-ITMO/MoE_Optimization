"""E1 ablation cell: MMLU x output_similarity grouping.

Run with ``python -m experts_fusion.run --config mmlu_output_similarity``.
"""
from experts_fusion.configs._base import *  # noqa: F401,F403 -- shared ablation surface
from experts_fusion.configs._base import __all__  # noqa: F401 -- re-export config surface
from common.names import PairingMethod

DATA_DIR = "./data/ef_mmlu_output_similarity"
PAIRING_METHOD = PairingMethod.OUTPUT_SIMILARITY
