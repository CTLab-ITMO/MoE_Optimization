"""Dispatch layer: loads one config from ``experts_fusion/configs/`` and
re-exports its values so all stages import ``experts_fusion.config``.

Select a config via ``run.py --config <name>`` (sets ``EXPERTS_FUSION_CONFIG``);
default is ``default``. Add configs by dropping a new ``configs/<name>.py``.
``config_snapshot_dict`` lives here so it always reads the live module globals.
"""
import importlib
import os

import torch

from common.fusion import FUSION_REGISTRY
from common.metrics import METRIC_REGISTRY

# Which config under experts_fusion/configs/ is active (run.py --config sets this).
_CONFIG_NAME = os.environ.get("EXPERTS_FUSION_CONFIG", "default")
_selected = importlib.import_module(f"experts_fusion.configs.{_CONFIG_NAME}")

# Re-export the selected config's public values into this module's namespace.
for _name in _selected.__all__:
    globals()[_name] = getattr(_selected, _name)

# Environment-derived (not part of a saved config); dropped from the snapshot.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DATA_DIR is excluded from the snapshot: it's the output location, not artifact
# content, so changing it is not "config drift". Every other field that affects
# produced artifacts must be in config_snapshot_dict (enforced by tests).
_SNAPSHOT_EXCLUDED = {"DATA_DIR"}


def config_snapshot_dict() -> dict:
    """Returns the active config as a plain dict for provenance snapshotting.

    Stage 1 writes this to ``config_snapshot.json``; later stages compare against
    it and fail fast on drift. Reads module globals at call time, so monkeypatching
    a config value in tests is reflected here automatically.
    """
    return {
        "MODEL_NAME": MODEL_NAME,
        "MODEL_DTYPE": MODEL_DTYPE,
        "EXPERTS": EXPERTS,
        "TRAIN_DATASET_NAME": TRAIN_DATASET_NAME,
        "VALIDATION_DATASET_NAME": VALIDATION_DATASET_NAME,
        "TEST_DATASET_NAME": TEST_DATASET_NAME,
        "DATASET_TEXT_FIELD": DATASET_TEXT_FIELD,
        "TRAIN_SIZE": TRAIN_SIZE,
        "VALIDATION_SIZE": VALIDATION_SIZE,
        "TEST_SIZE": TEST_SIZE,
        "LAYERS": LAYERS,
        "TOP_K": TOP_K,
        "SEED": SEED,
        "PAIRING_METHOD": PAIRING_METHOD,
        # Resolve None ("all") to the actual registered names for the snapshot.
        "FILTER_FUSIONS": FILTER_FUSIONS if FILTER_FUSIONS is not None else FUSION_REGISTRY.names(),
        "FILTER_METRICS": FILTER_METRICS if FILTER_METRICS is not None else METRIC_REGISTRY.names(),
        "USE_DEFAULT_EXPERTS": USE_DEFAULT_EXPERTS,
        "MIN_PAIR_VALIDATION_SAMPLES": MIN_PAIR_VALIDATION_SAMPLES,
        "INSPECT_FUSION_METHOD": INSPECT_FUSION_METHOD,
        "REPLACE_LAYERS_SEQUENTIAL": REPLACE_LAYERS_SEQUENTIAL,
        "REPLACE_BEST_LAYERS_SEQUENTIAL": REPLACE_BEST_LAYERS_SEQUENTIAL,
        "REPLACE_LAYERS_SMART_SEQUENTIAL": REPLACE_LAYERS_SMART_SEQUENTIAL,
        "ROUTER_ADAPTER": ROUTER_ADAPTER,
        "EXPERT_ADAPTER": EXPERT_ADAPTER,
        "QA_PARSER": QA_PARSER,
    }
