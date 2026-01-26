#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
import sys
import transformers
import datasets
from config import LAYERS, MODEL_NAME, TEST_DATASET_NAME, TEST_SIZE

sys.path.append("..")
from common.fusion_layer import get_fusion_layer_by_index


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

EXAMPLES = 10


# In[2]:


if "model" not in globals():
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_NAME)
if "tokenizer" not in globals():
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
model = model.to(device)


# In[3]:


if "_test_data" not in globals():
    _test_data = datasets.load_dataset(*TEST_DATASET_NAME, split="test")["question"]
test_data = _test_data[:EXAMPLES]


# In[4]:


def generate_examples(model):
	res = []
	for ind in range(EXAMPLES):
		messages = [
			{"role": "user", "content": test_data[ind]},
		]
		inputs = tokenizer.apply_chat_template(
			messages,
			add_generation_prompt=True,
			tokenize=True,
			return_dict=True,
			return_tensors="pt",
		).to(model.device)

		outputs = model.generate(**inputs, max_new_tokens=40)
		res.append((tokenizer.decode(outputs[0])))
	return res


# In[5]:


base_examples = generate_examples(model)

for layer in LAYERS:
    new_mlp = get_fusion_layer_by_index(layer, device=device).to(device)
    model.model.layers[layer].mlp = new_mlp

new_examples = generate_examples(model)


# In[9]:


for base_generation, new_generation in zip(base_examples, new_examples):
    print(base_generation, "\n" + "+" * 100 + "\n", new_generation)

