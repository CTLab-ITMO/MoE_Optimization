"""Default experiment config.

The values a run needs. Copy this file to ``experts_fusion/configs/<name>.py``,
edit it (e.g. a different ``DATA_DIR`` + dataset), and run it with
``python -m experts_fusion.run --config <name>`` -- several configs can run on
one disk without colliding because every output is scoped under ``DATA_DIR``.

``experts_fusion.config`` selects the active one of these by name (env var
``EXPERTS_FUSION_CONFIG``, set by ``run.py --config``; defaults to ``default``)
and re-exports the names in ``__all__`` below.
"""
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

# Column name holding the input text in the datasets above (read by stages 1 and 4).
DATASET_TEXT_FIELD = "question"

TRAIN_SIZE = 1_024       # reconstruction train + order-selection QA
VALIDATION_SIZE = 1_024  # reconstruction validation (held-out)
TEST_SIZE = 15_000       # final accuracy QA, large for tight CI
LAYERS = list(range(24))
TOP_K = 4

REPLACE_LAYERS_SEQUENTIAL = False
REPLACE_BEST_LAYERS_SEQUENTIAL = True
REPLACE_LAYERS_SMART_SEQUENTIAL = False

# Stage 4 (evaluate): keep original experts in memory next to fused ones for A/B
# comparison (True) or load only fused experts for production-style evaluation (False).
USE_DEFAULT_EXPERTS = False

# A group's fusion metrics are computed only when it has strictly MORE than this
# many validation activations (stage 3), i.e. the effective minimum is this + 1.
MIN_PAIR_VALIDATION_SAMPLES = 10

# Fusion method whose saved experts the stage-4 diagnostics (recon benchmark +
# generation) load and inspect. Must be one of the method names produced by stage 3.
INSPECT_FUSION_METHOD = "ZipIt"

SEED = 42

# Model/tokenizer adapters (dependency injection). Each hides a
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

FILTER_FUSIONS = None
FILTER_METRICS = None

# The config names re-exported by experts_fusion.config (the run-config surface).
__all__ = [
    "DATA_DIR",
    "MODEL_NAME",
    "MODEL_DTYPE",
    "EXPERTS",
    "TRAIN_DATASET_NAME",
    "VALIDATION_DATASET_NAME",
    "TEST_DATASET_NAME",
    "DATASET_TEXT_FIELD",
    "TRAIN_SIZE",
    "VALIDATION_SIZE",
    "TEST_SIZE",
    "LAYERS",
    "TOP_K",
    "REPLACE_LAYERS_SEQUENTIAL",
    "REPLACE_BEST_LAYERS_SEQUENTIAL",
    "REPLACE_LAYERS_SMART_SEQUENTIAL",
    "USE_DEFAULT_EXPERTS",
    "MIN_PAIR_VALIDATION_SAMPLES",
    "INSPECT_FUSION_METHOD",
    "SEED",
    "ROUTER_ADAPTER",
    "EXPERT_ADAPTER",
    "QA_PARSER",
    "PAIRING_METHOD",
    "FILTER_FUSIONS",
    "FILTER_METRICS",
]
