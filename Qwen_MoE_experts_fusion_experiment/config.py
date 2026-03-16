import torch


MODEL_NAME = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
EXPERTS = 60
TRAIN_DATASET_NAME = {"path": "cais/mmlu", "name": "all", "split": "auxiliary_train"}
TEST_DATASET_NAME = {"path": "commonsense_qa", "split": "test"}

TRAIN_SIZE = 1_000
TEST_SIZE = TRAIN_SIZE
VALIDATION_SIZE = 500
LAYERS = [0, 2, 3, 4, 5, 7, 8, 9, 10]
TOP_K = 4

EVAL_MODEL_AFTER_EVERY_STEP = True

device = torch.device("cuda")
