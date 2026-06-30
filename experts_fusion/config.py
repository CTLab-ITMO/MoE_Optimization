"""Active run config: selects one module under ``experts_fusion/configs/`` and
re-exports its values, so the stages keep importing ``experts_fusion.config``.

Pick the config by name with ``run.py --config <name>`` (which sets the
``EXPERTS_FUSION_CONFIG`` env var); the default is ``default``. Add a config by
dropping a new ``configs/<name>.py`` next to ``default.py``.

``config_snapshot_dict`` lives here (not in the selected config) so it reads
*these* module globals -- the ones the stages and the test harness see and
monkeypatch.
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

# Config fields deliberately kept OUT of the snapshot: DATA_DIR is the output
# *location* (the knob for running several experiments on one disk), not artifact
# content, so two runs at different DATA_DIRs are not "config drift". Everything
# else that affects produced artifacts must appear in config_snapshot_dict -- a
# test (test_experts_fusion_config) enforces this so a new field can't silently
# escape drift detection.
_SNAPSHOT_EXCLUDED = {"DATA_DIR"}


def config_snapshot_dict() -> dict:
    """The run config as a plain value map for the provenance snapshot.

    Stage 1 writes this canonicalized (``common.provenance``: ``device`` dropped,
    dtype/adapter values stabilized to names) to ``config_snapshot.json``; every
    later stage compares the live config against it and fails fast on drift. The
    function reads the module globals at call time, so a test that monkeypatches
    a config attribute is reflected here. The config serializes itself -- the
    library code in ``common`` never imports this module.
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
        # Resolve ``None`` (= "all") to the concrete registered set, so the
        # snapshot records exactly which methods/metrics a run used.
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
