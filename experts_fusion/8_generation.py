import torch
from tqdm import tqdm
import transformers
import datasets
from experts_fusion.config import (
    LAYERS,
    EXPERTS,
    MODEL_NAME,
    MODEL_DTYPE,
    TOP_K,
    VALIDATION_DATASET_NAME,
    DATASET_TEXT_FIELD,
    INSPECT_FUSION_METHOD,
    ROUTER_ADAPTER,
    device,
    DATA_DIR,
)

from common import layout
from common.fusion_layer import get_fusion_layer_by_index
from common.model_topology import discover_moe_layers

EXAMPLES = 10
# Number of tokens generated per prompt in the before/after qualitative comparison.
MAX_NEW_TOKENS = 10


from common.seeding import set_seed
from experts_fusion.config import SEED
set_seed(SEED)


if "model" not in globals():
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=MODEL_DTYPE)
if "tokenizer" not in globals():
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
model = model.to(device)


if "_validation_data" not in globals():
    _validation_data = datasets.load_dataset(**VALIDATION_DATASET_NAME)[DATASET_TEXT_FIELD]
validation_data = _validation_data[:EXAMPLES]


@torch.autocast(device.type)
def generate_examples(model, examples):
    res = []
    for example in tqdm(examples):
        outputs = model.generate(**example, max_new_tokens=MAX_NEW_TOKENS)
        res.append((tokenizer.decode(outputs[0])))
    return res


examples = []
for ind in range(EXAMPLES):
    messages = [
        {"role": "user", "content": validation_data[ind]},
    ]
    examples.append(tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device))


base_examples = generate_examples(model, examples)


for layer in discover_moe_layers(model, selected=LAYERS):
    del model.model.layers[layer].mlp
    torch.cuda.empty_cache()
    new_mlp = get_fusion_layer_by_index(
        layer,
        layout.fusion_method_dir(INSPECT_FUSION_METHOD),
        DATA_DIR,
        device=device,
        use_default_experts=False,
        num_experts=EXPERTS,
        router_adapter=ROUTER_ADAPTER,
        top_k=TOP_K,
    ).to(device)
    model.model.layers[layer].mlp = new_mlp


new_examples = generate_examples(model, examples)


for base_generation, new_generation in zip(base_examples, new_examples):
    print(
        base_generation,
        "\n" + "+" * 100 + "\n",
        new_generation,
        "\n" + "-" * 100 + "\n"
    )

