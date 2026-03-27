import torch


MODEL_NAME = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
EXPERTS = 60
TRAIN_DATASET_NAME = {"path": "commonsense_qa", "split": "train"}
TEST_DATASET_NAME = {"path": "commonsense_qa", "split": "test"}

TRAIN_SIZE = 15_000
TEST_SIZE = TRAIN_SIZE
VALIDATION_SIZE = 1_000
LAYERS = list(range(24))
TOP_K = 4

EVAL_MODEL_AFTER_EVERY_STEP = True
EVOLUTION_STEPS = 50

device = torch.device("cuda")