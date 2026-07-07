"""Default experiment config for the experts-fusion pipeline.

To run a different experiment, copy this file to ``experts_fusion/configs/<name>.py``,
adjust the values, and pass ``--config <name>`` to ``python -m experts_fusion.run``.
Each config scopes its outputs under its own ``DATA_DIR``, so multiple configs coexist
on one disk without collision. The active config is selected by the
``EXPERTS_FUSION_CONFIG`` env var (set by ``run.py --config``; defaults to ``default``).
"""
import torch

from common.expert_adapter import Qwen15ExpertAdapter, Qwen15SharedExpertAdapter
from common.fusion import ZipItFusion
from common.names import PairingMethod
from common.qa_parser import Qwen15QAParser
from common.router import Qwen15RouterAdapter

DATA_DIR = "./data/data_experts_fusion"

MODEL_NAME = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
MODEL_DTYPE = torch.bfloat16
EXPERTS = 60
TRAIN_DATASET_NAME = {"path": "cais/mmlu", "name": "all", "split": "auxiliary_train"}
VALIDATION_DATASET_NAME = {"path": "cais/mmlu", "name": "all", "split": "validation"}
TEST_DATASET_NAME = {"path": "cais/mmlu", "name": "all", "split": "test"}

# Input text field in the datasets above; read by stages 1 and 4.
DATASET_TEXT_FIELD = "question"

TRAIN_SIZE = 1_024       # train set for reconstruction and layer-order selection
VALIDATION_SIZE = 4_096  # held-out reconstruction-validation set; raised from 1_024 per E4 decision #5 (caps at the split size)
TEST_SIZE = 15_000       # large set for final accuracy evaluation
LAYERS = list(range(24))
TOP_K = 4

REPLACE_LAYERS_SEQUENTIAL = False
REPLACE_BEST_LAYERS_SEQUENTIAL = True
REPLACE_LAYERS_SMART_SEQUENTIAL = False

# True: keep original experts alongside fused ones for A/B comparison.
# False: load only fused experts (production mode).
USE_DEFAULT_EXPERTS = False

# Stage 3: compute fusion metrics only for groups with more than this many
# validation activations (effective minimum = this value + 1).
MIN_PAIR_VALIDATION_SAMPLES = 10

# Method whose fused experts stage 4 uses for diagnostics (recon + generation).
# Must be one of the method names produced by stage 3.
INSPECT_FUSION_METHOD = ZipItFusion.name

SEED = 42

# Dependency-injection adapters for model-family-specific logic. Swap the Qwen15*
# instances to retarget the pipeline at another model family without touching the stages.
#   - ROUTER_ADAPTER: gate output → (selected_experts, routing_weights).
#   - EXPERT_ADAPTER: extracts each expert as an AdvancedMLP (stage 1).
#   - SHARED_EXPERT_ADAPTER: extracts (stage 1) and rebuilds (stage 4) the shared
#     expert; NoSharedExpertAdapter is the fallback for families without one.
#   - QA_PARSER: builds eval prompts and resolves answer token ids (stage 4).
ROUTER_ADAPTER = Qwen15RouterAdapter()
EXPERT_ADAPTER = Qwen15ExpertAdapter()
SHARED_EXPERT_ADAPTER = Qwen15SharedExpertAdapter()
QA_PARSER = Qwen15QAParser()

# Expert-pairing strategy; one of: activation_correlation, random, output_similarity.
PAIRING_METHOD = PairingMethod.ACTIVATION_CORRELATION

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
    "SHARED_EXPERT_ADAPTER",
    "QA_PARSER",
    "PAIRING_METHOD",
    "FILTER_FUSIONS",
    "FILTER_METRICS",
]
