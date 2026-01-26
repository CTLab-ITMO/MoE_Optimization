MODEL_NAME = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
EXPERTS = 60
TRAIN_DATASET_NAME = "openai/gsm8k", "main"
TEST_DATASET_NAME = "cais/mmlu", "abstract_algebra"

TRAIN_SIZE = 100_000
TEST_SIZE = TRAIN_SIZE
LAYERS = [0]
TOP_K = 8
