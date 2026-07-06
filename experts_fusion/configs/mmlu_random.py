"""E1 ablation cell: MMLU x random grouping (the ablation's lower bound).

Run with ``python -m experts_fusion.run --config mmlu_random``.
"""
from experts_fusion.configs._base import *  # noqa: F401,F403 -- shared ablation surface
from experts_fusion.configs._base import __all__  # noqa: F401 -- re-export config surface
from common.names import PairingMethod

DATA_DIR = "./data/ef_mmlu_random"
PAIRING_METHOD = PairingMethod.RANDOM
