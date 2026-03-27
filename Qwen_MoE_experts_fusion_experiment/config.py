import torch

DATA_DIR = "./data"

MODEL_NAME = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
EXPERTS = 60
TRAIN_DATASET_NAME = {"path": "commonsense_qa", "split": "train"}
TEST_DATASET_NAME = {"path": "commonsense_qa", "split": "test"}

TRAIN_SIZE = 15_000
TEST_SIZE = TRAIN_SIZE
VALIDATION_SIZE = 1_000
LAYERS = [16, 17, 22, 14, 19,21, 18, 20, 23, 5, 15, 3, 2, 10, 11, 13, 4, 8, 12, 9, 7, 0, 1, 6]
TOP_K = 4

REPLACE_LAYERS_SEQUANTIAL = False
EVOLUTION_STEPS = 50

device = torch.device("cuda")
