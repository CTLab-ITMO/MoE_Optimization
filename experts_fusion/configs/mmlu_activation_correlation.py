"""E1 ablation cell: MMLU x activation_correlation grouping.

The base module already defaults to this cell; only DATA_DIR is scoped here so the
cell's outputs never collide with the other five. Run with
``python -m experts_fusion.run --config mmlu_activation_correlation``.
"""
from experts_fusion.configs._base import *  # noqa: F401,F403 -- shared ablation surface
from experts_fusion.configs._base import __all__  # noqa: F401 -- re-export config surface

DATA_DIR = "./data/ef_mmlu_activation_correlation"
