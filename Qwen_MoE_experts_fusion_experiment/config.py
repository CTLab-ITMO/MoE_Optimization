import torch

DATA_DIR = "./data_MMLU"

MODEL_NAME = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
EXPERTS = 60
TRAIN_DATASET_NAME = {"path": "cais/mmlu", "name": "all", "split": "auxiliary_train"}
TEST_DATASET_NAME = {"path": "cais/mmlu", "name": "all", "split": "test"}

TRAIN_SIZE = 15_000
TEST_SIZE = TRAIN_SIZE
VALIDATION_SIZE = 1_000
LAYERS = list(range(24))
TOP_K = 4

REPLACE_LAYERS_SEQUANTIAL = False
REPLACE_BEST_LAYERS_SEQUANTIAL = True
REPLACE_LAYERS_SMART_SEQUANTIAL = True
EVOLUTION_STEPS = 50

device = torch.device("cuda")

FILTER_FUSIONS = None
FILTER_METRICS = None
