import torch
import transformers

DATA_DIR = "./data_MMLU_1k"

MODEL_NAME = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
EXPERTS = 60
MODEL_NAME = "deepseek-ai/DeepSeek-V2-Lite-Chat"
EXPERTS = 64
TRAIN_DATASET_NAME = {"path": "cais/mmlu", "name": "all", "split": "auxiliary_train"}
TEST_DATASET_NAME = {"path": "cais/mmlu", "name":"all", "split": "test"}

TRAIN_SIZE = 128 #1_024 changed glazkov
TEST_SIZE = TRAIN_SIZE
VALIDATION_SIZE = 1_000
LAYERS = list(range(24)) #BE CAREFUL

#in deepseek 0 layer don't contain any expert.
#we want to avoid finding gate and experts in it
def is_deepseek_experiment():
    return 'deepseek' in MODEL_NAME.lower()

if (is_deepseek_experiment()):
    layers2 = []
    for layer_index in LAYERS:
        if (layer_index != 0):
            layers2.append(layer_index)
    
    LAYERS = layers2

TOP_K = 4

REPLACE_LAYERS_SEQUANTIAL = False
REPLACE_BEST_LAYERS_SEQUANTIAL = True
REPLACE_LAYERS_SMART_SEQUANTIAL = False
EVOLUTION_STEPS = 50

SEED = 42

# "activation_correlation", "random", "output_similarity"
PAIRING_METHOD = "create_clusters_by_EC_greedy_aglomerative" #"activation_correlation"

device = torch.device("cuda:3")
# device = "auto"

FILTER_FUSIONS = ["FineTune_ZipIt"]
FILTER_METRICS = []

COMPARE_WITH_ZERO_FUSION_IN_9_SCRIPT = False

def correct_model_loading():
    if (is_deepseek_experiment()):
        from transformers.cache_utils import DynamicCache

        # DynamicCache.get_usable_length = DynamicCache.get_seq_length #fix for deepseek!!!
        
        quantization_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,  # Compute in float16 for speed
            bnb_4bit_use_double_quant=True,  # Double quantization for extra memory savings
            # Normalized float 4-bit (optimal for LLMs)
            bnb_4bit_quant_type="nf4"
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, 
            trust_remote_code=True,
            device_map=device,
            cache_dir='/glazkov-dev/.cache',
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    else: #assume it's qwen
        model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True, device_map=device, dtype=torch.float16)
        tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    return model, tokenizer