import torch

DATA_DIR = "./data_MMLU_1k"

MODEL_NAME = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
MODEL_DTYPE = torch.bfloat16
EXPERTS = 60
TRAIN_DATASET_NAME = {"path": "cais/mmlu", "name": "all", "split": "auxiliary_train"}
VALIDATION_DATASET_NAME = {"path": "cais/mmlu", "name": "all", "split": "validation"}
TEST_DATASET_NAME = {"path": "cais/mmlu", "name": "all", "split": "test"}

TRAIN_SIZE = 1_024       # reconstruction train + stage-9 order-selection QA
VALIDATION_SIZE = 1_024  # reconstruction validation (held-out, stages 1-8)
TEST_SIZE = 15_000       # final accuracy QA (stage 9), large for tight CI
LAYERS = list(range(24))
TOP_K = 4

REPLACE_LAYERS_SEQUANTIAL = False
REPLACE_BEST_LAYERS_SEQUANTIAL = True
REPLACE_LAYERS_SMART_SEQUANTIAL = False
EVOLUTION_STEPS = 50

SEED = 42

# "activation_correlation", "random", "output_similarity"
PAIRING_METHOD = "activation_correlation"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FILTER_FUSIONS = None
FILTER_METRICS = None
