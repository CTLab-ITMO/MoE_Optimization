import torch

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
EVOLUTION_STEPS = 50

# Stage 9: keep original experts in memory next to fused ones for A/B comparison
# (True) or load only fused experts for production-style evaluation (False).
USE_DEFAULT_EXPERTS = False

# Minimum number of validation activations a pair must have before stage 6 computes
# and records its fusion metrics.
MIN_PAIR_VALIDATION_SAMPLES = 10

# Fusion method whose saved experts the qualitative stages (7 benchmark, 8 generation)
# load and inspect. Must be one of the method names produced by stage 6.
INSPECT_FUSION_METHOD = "ZipIt"

SEED = 42

# "activation_correlation", "random", "output_similarity"
PAIRING_METHOD = "activation_correlation"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FILTER_FUSIONS = None
FILTER_METRICS = None
