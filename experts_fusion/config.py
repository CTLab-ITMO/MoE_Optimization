import torch

from common.expert_adapter import Qwen15ExpertAdapter
from common.qa_parser import Qwen15QAParser
from common.router import Qwen15RouterAdapter

DATA_DIR = "./data/data_experts_fusion"

MODEL_NAME = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
MODEL_DTYPE = torch.bfloat16
EXPERTS = 60
TRAIN_DATASET_NAME = {"path": "cais/mmlu", "name": "all", "split": "auxiliary_train"}
VALIDATION_DATASET_NAME = {"path": "cais/mmlu", "name": "all", "split": "validation"}
TEST_DATASET_NAME = {"path": "cais/mmlu", "name": "all", "split": "test"}

# Column name holding the input text in the datasets above (read by stages 1, 4, 8).
DATASET_TEXT_FIELD = "question"

TRAIN_SIZE = 1_024       # reconstruction train + stage-9 order-selection QA
VALIDATION_SIZE = 1_024  # reconstruction validation (held-out, stages 1-8)
TEST_SIZE = 15_000       # final accuracy QA (stage 9), large for tight CI
LAYERS = list(range(24))
TOP_K = 4

REPLACE_LAYERS_SEQUANTIAL = False
REPLACE_BEST_LAYERS_SEQUANTIAL = True
REPLACE_LAYERS_SMART_SEQUANTIAL = False

# Stage 4 (evaluate): keep original experts in memory next to fused ones for A/B
# comparison (True) or load only fused experts for production-style evaluation (False).
USE_DEFAULT_EXPERTS = False

# Minimum number of validation activations a group must have before stage 3
# (fusion) computes and records its fusion metrics.
MIN_PAIR_VALIDATION_SAMPLES = 10

# Fusion method whose saved experts the stage-4 diagnostics (recon benchmark +
# generation) load and inspect. Must be one of the method names produced by stage 3.
INSPECT_FUSION_METHOD = "ZipIt"

SEED = 42

# Model/tokenizer adapters (dependency injection, §3.2). Each hides a
# model-family-specific concern behind a common/ interface; the Qwen15* defaults
# reproduce the current Qwen1.5-MoE behavior. Swap the instance to retarget the
# pipeline at another family without editing the stages.
#   - ROUTER_ADAPTER: canonicalizes the gate output into (selected_experts,
#     routing_weights); normalize_topk=False matches Qwen1.5 norm_topk_prob=False.
#   - EXPERT_ADAPTER: extracts each routed expert as an AdvancedMLP (stage 1).
#   - QA_PARSER: builds the eval prompt + resolves answer-letter token ids (stage 4).
ROUTER_ADAPTER = Qwen15RouterAdapter()
EXPERT_ADAPTER = Qwen15ExpertAdapter()
QA_PARSER = Qwen15QAParser()

# "activation_correlation", "random", "output_similarity"
PAIRING_METHOD = "activation_correlation"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FILTER_FUSIONS = None
FILTER_METRICS = None

# Opt-in discovery of installed third-party plugins (fusion/grouping/metric/schema)
# advertised via entry points. Off by default so the in-tree registries stay
# pinned; set True to let an installed plugin take effect.
LOAD_PLUGINS = False

# Config fields deliberately kept OUT of the snapshot: DATA_DIR is the output
# *location* (the knob for running several experiments on one disk), not artifact
# content, so two runs at different DATA_DIRs are not "config drift". Everything
# else that affects produced artifacts must appear in config_snapshot_dict -- a
# test (test_experts_fusion_config) enforces this so a new field can't silently
# escape drift detection.
_SNAPSHOT_EXCLUDED = {"DATA_DIR"}


def config_snapshot_dict() -> dict:
    """The run config as a plain value map for the provenance snapshot (§3.6).

    Stage 1 writes this canonicalized (``common.provenance``: ``device`` dropped,
    dtype/adapter values stabilized to names) to ``config_snapshot.json``; every
    later stage compares the live config against it and fails fast on drift. The
    function reads the module globals at call time, so a test that monkeypatches
    a config attribute is reflected here. The config serializes itself -- the
    library code in ``common`` never imports this module (§1.3).
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
        "FILTER_FUSIONS": FILTER_FUSIONS,
        "FILTER_METRICS": FILTER_METRICS,
        "USE_DEFAULT_EXPERTS": USE_DEFAULT_EXPERTS,
        "MIN_PAIR_VALIDATION_SAMPLES": MIN_PAIR_VALIDATION_SAMPLES,
        "INSPECT_FUSION_METHOD": INSPECT_FUSION_METHOD,
        "REPLACE_LAYERS_SEQUANTIAL": REPLACE_LAYERS_SEQUANTIAL,
        "REPLACE_BEST_LAYERS_SEQUANTIAL": REPLACE_BEST_LAYERS_SEQUANTIAL,
        "REPLACE_LAYERS_SMART_SEQUANTIAL": REPLACE_LAYERS_SMART_SEQUANTIAL,
        "ROUTER_ADAPTER": ROUTER_ADAPTER,
        "EXPERT_ADAPTER": EXPERT_ADAPTER,
        "QA_PARSER": QA_PARSER,
        "LOAD_PLUGINS": LOAD_PLUGINS,
    }
