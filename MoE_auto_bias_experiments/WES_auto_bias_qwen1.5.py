import os
import random
import re
import warnings
from collections import defaultdict
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.set_verbosity_error()

# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6"

SEED = 1235
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


print("[DEBUG] Загрузка датасетов MMLU...")
# Load MMLU datasets
mmlu_dataset = load_dataset("cais/mmlu", "all")
test_samples_all = mmlu_dataset["test"].shuffle(seed=SEED).select(range(300))

mmlu_dataset = load_dataset("cais/mmlu", "abstract_algebra")
test_samples_alg = mmlu_dataset["test"].shuffle(seed=SEED).select(range(100))

mmlu_dataset = load_dataset("cais/mmlu", "college_mathematics")
test_samples_math = mmlu_dataset["test"].shuffle(seed=SEED).select(range(100))
print(f"[DEBUG] Датысеты загружены: math={len(test_samples_math)}, alg={len(test_samples_alg)}, all={len(test_samples_all)}")


# abstract_algebra -> Abstract Algebra
def pretty_subject(s: str) -> str:
    return s.replace("_", " ").title()

def format_mmlu_question(example):
    subject = pretty_subject(example.get("subject"))
    question = example["question"]
    choices = example["choices"]
    return (
        f"The following is a multiple choice question in {subject}.\n"
        f"Choose the correct answer from the options.\n\n"
        f"Question: {question}\n\n"
        f"Options:\n"
        f"A. {choices[0]}\n"
        f"B. {choices[1]}\n"
        f"C. {choices[2]}\n"
        f"D. {choices[3]}\n\n"
        f"Answer:"
)


def generate_response(model, tokenizer, prompt):
    #print(f"[DEBUG] Генерация ответа модели...")
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    #print(f"[DEBUG] Токенизированный промпт (длина: {model_inputs['input_ids'].shape[1]})")

    with torch.no_grad():
        #print(f"[DEBUG] Начало генерации...")
        outputs = model.generate(**model_inputs, max_new_tokens=10, do_sample=False)
        #print(f"[DEBUG] Генерация завершена (выходная длина: {outputs.shape[1]})")

    response = tokenizer.decode(outputs[0][model_inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
    #print(f"[DEBUG] Декодированный ответ: '{response}'")
    return response.strip()


def extract_answer(text: str):
    #print(f"[DEBUG] Извлечение ответа из текста: '{text}'")
    if not text:
        #print(f"[DEBUG] Пустой текст, возвращаю None")
        return None

    t_raw = text.strip()
    t = t_raw.upper()

    # 0) Самые частые шаблоны: "The correct answer is C", "Answer is: D", "Answer: B", "Option C"
    # Ищем букву A-D после ключевых слов/фраз.
    patterns = [
        r"\bTHE\s+CORRECT\s+ANSWER\s+IS\s*[:\-]?\s*[\(\[]?\s*([ABCD])\s*[\)\]]?\b",
        r"\bCORRECT\s+ANSWER\s+IS\s*[:\-]?\s*[\(\[]?\s*([ABCD])\s*[\)\]]?\b",
        r"\bANSWER\s+IS\s*[:\-]?\s*[\(\[]?\s*([ABCD])\s*[\)\]]?\b",
        r"\bANSWER\s*[:\-]\s*[\(\[]?\s*([ABCD])\s*[\)\]]?\b",
        #r"\bOPTION\s*[:\-]?\s*[\(\[]?\s*([ABCD])\s*[\)\]]?\b",
        #r"\bCHOICE\s*[:\-]?\s*[\(\[]?\s*([ABCD])\s*[\)\]]?\b",
        #r"\bSELECT\s+([ABCD])\b",
        #r"\bIS\s+([ABCD])\b",  # слабее, но иногда "It is C."
    ]
    for p in patterns:
        m = re.search(p, t)
        if m:
            result = m.group(1)
            #print(f"[DEBUG] Найден ответ (шаблон): {result} по паттерну {p}")
            return result

    # 1) В начале: "A", "A.", "A)", "A:" ...
    m = re.match(r"^\s*([ABCD])(?:[\.\)\:\-]|$)", t)
    if m:
        result = m.group(1)
        #print(f"[DEBUG] Найден ответ (начало строки): {result}")
        return result

    # 2) Первая standalone буква где угодно
    m_first = re.search(r"\b([ABCD])\b", t)
    if m_first:
        result = m_first.group(1)
        #print(f"[DEBUG] Найден ответ (первая standalone буква): {result}")
        return result

    # 3) Последняя standalone буква (на всякий)
    m_all = re.findall(r"\b([ABCD])\b", t)
    result = m_all[-1] if m_all else None
    #print(f"[DEBUG] Найден ответ (последняя standalone буква): {result}")
    return result


def test_accuracy(model, tokenizer, test_samples):
    correct = 0
    print(f"[DEBUG] Тестирование точности на {len(test_samples)} примерах")

    for i, example in enumerate(tqdm(test_samples, desc="Testing accuracy", total=len(test_samples))):
        #print(f"[DEBUG] Пример {i+1}/{len(test_samples)}")
        prompt = format_mmlu_question(example)
        response = generate_response(model, tokenizer, prompt)
        predicted = extract_answer(response)
        correct_answer = ["A", "B", "C", "D"][example["answer"]]

        #print(f"[DEBUG] Предсказано: {predicted}, Правильный ответ: {correct_answer}")

        if predicted == correct_answer:
            correct += 1
            #print(f"[DEBUG] ✓ Правильно")
        # else:
            #print(f"[DEBUG] ✗ Неправильно")

    accuracy = correct / len(test_samples)
    print(f"[DEBUG] Итоговая точность: {correct}/{len(test_samples)} = {accuracy:.4f}")
    return accuracy

def compute_expert_importance(model, tokenizer, dataset):
    print(f"[DEBUG] Вычисление важности экспертов на {len(dataset)} примерах")
    model.eval()
    num_layers = len(model.model.layers)
    print(f"[DEBUG] Модель имеет {num_layers} слоев")

    expert_importance = [defaultdict(float) for _ in range(num_layers)]
    all_router_logits = defaultdict(list)
    expert_outputs_sum = [{} for _ in range(num_layers)]
    expert_outputs_count = [defaultdict(int) for _ in range(num_layers)]

    for i, example in enumerate(tqdm(dataset, desc="Computing expert importance", total=len(dataset))):
        #print(f"[DEBUG] Пример {i+1}/{len(dataset)}: Вычисление важности экспертов")

        prompt = format_mmlu_question(example)
        correct_answer = ["A", "B", "C", "D"][example["answer"]]
        #print(f"[DEBUG] Правильный ответ: {correct_answer}")

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        #print(f"[DEBUG] Токенизированный промпт (длина: {inputs['input_ids'].shape[1]})")

        answer_text = f" {correct_answer}"
        answer_ids = tokenizer(answer_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
        #print(f"[DEBUG] Токены ответа: {answer_ids.shape}")

        prompt_len = inputs["input_ids"].shape[1]
        full_ids = torch.cat([inputs["input_ids"], answer_ids], dim=1)
        #print(f"[DEBUG] Полные токены (длина: {full_ids.shape[1]}, промпт: {prompt_len}, ответ: {answer_ids.shape[1]})")

        labels = torch.full_like(full_ids, -100)
        labels[:, prompt_len:] = answer_ids
        #print(f"[DEBUG] Labels созданы")

        hidden_states_with_grad = {}
        mlp_inputs = {}

        def make_backward_hook(layer_idx):
            def hook(module, grad_input, grad_output):
                if grad_output[0] is not None:
                    hidden_states_with_grad[layer_idx] = grad_output[0].clone()
                    #print(f"[DEBUG] Backward hook слой {layer_idx}: grad_output сохранен")
            return hook

        def make_forward_hook(layer_idx):
            def hook(module, input, output):
                mlp_inputs[layer_idx] = input[0].detach().clone()
                #print(f"[DEBUG] Forward hook слой {layer_idx}: mlp_input сохранен")
            return hook

        hooks = []
        #print(f"[DEBUG] Регистрация hooks...")
        for layer_idx in range(num_layers):
            backward_hook = model.model.layers[layer_idx].mlp.register_full_backward_hook(make_backward_hook(layer_idx))
            forward_hook = model.model.layers[layer_idx].mlp.register_forward_hook(make_forward_hook(layer_idx))
            hooks.append(backward_hook)
            hooks.append(forward_hook)
        #print(f"[DEBUG] Все hooks зарегистрированы ({len(hooks)} total)")

        #print(f"[DEBUG] Прямой проход (forward pass)...")
        outputs = model(input_ids=full_ids, labels=labels, output_hidden_states=True)
        loss = outputs.loss
        #print(f"[DEBUG] Прямой проход завершен, loss = {loss.item():.6f}")

        #print(f"[DEBUG] Обратный проход (backward pass)...")
        model.zero_grad()
        loss.backward()
        #print(f"[DEBUG] Обратный проход завершен, градиенты вычислены")

        #print(f"[DEBUG] Удаление hooks...")
        for hook in hooks:
            hook.remove()
        #print(f"[DEBUG] Hooks удалены")

        #print(f"[DEBUG] Анализ градиентов для каждого слоя...")
        for layer_idx in range(num_layers):
            if layer_idx not in hidden_states_with_grad:
                #print(f"[DEBUG] Слой {layer_idx}: нет градиентов, пропускаем")
                continue

            #print(f"[DEBUG] Слой {layer_idx}: анализ важности экспертов")
            layer = model.model.layers[layer_idx]
            g = hidden_states_with_grad[layer_idx]
            hidden_flat = mlp_inputs[layer_idx]

            batch_size, seq_len, hidden_dim = hidden_flat.shape
            hidden_flat = hidden_flat.view(-1, hidden_dim)
            g_flat = g.view(-1, hidden_dim).to(hidden_flat.device)
            #print(f"[DEBUG] Слой {layer_idx}: hidden_flat shape = {hidden_flat.shape}, g_flat shape = {g_flat.shape}")

            router_logits = layer.mlp.gate(hidden_flat)
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            top_k = layer.mlp.num_experts
            #print(f"[DEBUG] top-k is: {top_k}")

            weights, selected = torch.topk(routing_weights, top_k, dim=-1)
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-6)
            weights = weights.to(hidden_flat.device)
            selected = selected.to(hidden_flat.device)
            #print(f"[DEBUG] Слой {layer_idx}: top_k = {top_k}, weights shape = {weights.shape}")

            answer_start = prompt_len
            answer_end = prompt_len + answer_ids.shape[1]
            #print(f"[DEBUG] Слой {layer_idx}: answer tokens range = [{answer_start}, {answer_end})")

            for token_idx in range(answer_start, answer_end):
                grad_token_idx = token_idx - 1
                #print(f"[DEBUG] Слой {layer_idx}: токен {token_idx} (grad index: {grad_token_idx})")

                for k_idx in range(top_k):
                    expert_idx = selected[grad_token_idx, k_idx].item()
                    w_i = weights[grad_token_idx, k_idx].item()

                    with torch.no_grad():
                        h_i = layer.mlp.experts[expert_idx](hidden_flat[grad_token_idx : grad_token_idx + 1]).to(hidden_flat.device)
                        h_i_flat = h_i.squeeze(0).detach()
                        if expert_idx not in expert_outputs_sum[layer_idx]:
                            expert_outputs_sum[layer_idx][expert_idx] = torch.zeros_like(h_i_flat)
                        expert_outputs_sum[layer_idx][expert_idx] += h_i_flat
                        expert_outputs_count[layer_idx][expert_idx] += 1

                    g_token = g_flat[grad_token_idx]
                    importance = -torch.dot(g_token, (w_i * h_i).squeeze(0)).item()
                    expert_importance[layer_idx][expert_idx] += importance

                    # if i == 0 and token_idx == answer_start and k_idx == 0:
                        #print(f"[DEBUG] Слой {layer_idx}: expert {expert_idx}, w_i = {w_i:.4f}, importance = {importance:.6f}")

                all_router_logits[layer_idx].append(router_logits[token_idx])
        #print(f"[DEBUG] Сброс градиентов модели...")
        model.zero_grad()
        #print(f"[DEBUG] Пример {i+1} завершен")

    # Вычисляем средние активации
    expert_outputs_mean = [{} for _ in range(num_layers)]
    for l in range(num_layers):
        for e in expert_outputs_sum[l]:
            if expert_outputs_count[l][e] > 0:
                expert_outputs_mean[l][e] = (expert_outputs_sum[l][e] / expert_outputs_count[l][e]).cpu()

    #print(f"[DEBUG] Вычисление важности экспертов завершено")
    return expert_importance, all_router_logits, expert_outputs_mean

def apply_expert_bias_v1(model, expert_importance, all_router_logits, expert_outputs_mean=None):
    print(f"[DEBUG] Применение смещения (bias) к экспертам с eps = mean(|logits|) / num_experts")
    num_layers = len(model.model.layers)

    for layer_idx in range(num_layers):
        is_debug_layer = layer_idx in [1, 7, num_layers - 1]
        if is_debug_layer:
            print(f"[DEBUG] Обработка слоя {layer_idx}")
        
        layer = model.model.layers[layer_idx]

        if len(expert_importance[layer_idx]) == 0:
            if is_debug_layer:
                print(f"[DEBUG] Слой {layer_idx}: нет данных о важности экспертов, пропускаем")
            continue

        importances = torch.zeros(layer.mlp.num_experts)
        for expert_idx, imp in expert_importance[layer_idx].items():
            importances[expert_idx] = imp

        if is_debug_layer:
            print(f"[DEBUG] Слой {layer_idx}: важности до нормализации: {' '.join([f'{x:.4f}' for x in importances.tolist()])}")
            print(f"[DEBUG] Слой {layer_idx}: Mean (raw): {importances.mean().item():.4f}, Std (raw): {importances.std().item():.4f}")

        mean_imp = importances.mean()
        std_imp = importances.std()

        if std_imp > 1e-8:
            importances = (importances - mean_imp) / std_imp
            if is_debug_layer:
                print(f"[DEBUG] Слой {layer_idx}: важности после нормализации: {' '.join([f'{x:.4f}' for x in importances.tolist()])}")
                print(f"[DEBUG] Слой {layer_idx}: Mean (norm): {importances.mean().item():.4f}, Std (norm): {importances.std().item():.4f}")
        else:
            if is_debug_layer:
                print(f"[DEBUG] Слой {layer_idx}: std слишком мал ({std_imp:.6f}), пропускаем нормализацию")

        logit_means_per_sample = []
        
        # Печать активаций роутера (logits до softmax)
        all_logits_for_layer = torch.stack(all_router_logits[layer_idx])
        mean_router_logits = all_logits_for_layer.mean(dim=0)
        
        if is_debug_layer:
            print(f"[DEBUG] Слой {layer_idx}: router logit (pre-softmax) mean: {' '.join([f'{x:.4f}' for x in mean_router_logits.tolist()])}")
            print(f"[DEBUG] Слой {layer_idx}: router logit (pre-softmax) global mean: {mean_router_logits.mean().item():.4f}, global std: {mean_router_logits.std().item():.4f}")
            
            last_3_logits = all_router_logits[layer_idx][-3:]
            for i, token_logits in enumerate(last_3_logits):
                print(f"[DEBUG] Слой {layer_idx}: router logit (pre-softmax) token -{3-i}: {' '.join([f'{x:.4f}' for x in token_logits.flatten().tolist()])}")

        for router_logits in all_router_logits[layer_idx]:
            logit_means_per_sample.append(router_logits.abs().mean().item())

        logit_means_per_dataset = np.mean(logit_means_per_sample)

        with torch.no_grad():
            if layer.mlp.gate.bias is None:
                num_experts = layer.mlp.num_experts
                layer.mlp.gate.bias = nn.Parameter(torch.zeros(num_experts, dtype=layer.mlp.gate.weight.dtype, device=layer.mlp.gate.weight.device))
                if is_debug_layer:
                    print(f"[DEBUG] Слой {layer_idx}: создан новый bias параметр (size={num_experts})")
            
            eps = logit_means_per_dataset / layer.mlp.num_experts
            
            old_bias = layer.mlp.gate.bias.clone()
            layer.mlp.gate.bias += eps * importances.to(layer.mlp.gate.bias.device)
            
            if is_debug_layer:
                print(f"[DEBUG] Слой {layer_idx}: eps = {eps}")
                bias_change = torch.norm(layer.mlp.gate.bias - old_bias).item()
                print(f"[DEBUG] Слой {layer_idx}: bias обновлен (изменение нормы: {bias_change:.6f})")
                print(f"[DEBUG] Слой {layer_idx}: Итоговый bias: {' '.join([f'{x:.4f}' for x in layer.mlp.gate.bias.tolist()])}")

                for exp_idx in range(len(layer.mlp.experts)):
                    if expert_outputs_mean and exp_idx in expert_outputs_mean[layer_idx]:
                        act_vals = expert_outputs_mean[layer_idx][exp_idx].flatten()[:20].tolist()
                        print(f"[DEBUG] Expert {exp_idx} mean output activations (h_i): {' '.join([f'{x:.4f}' for x in act_vals])}")

                print(f"[DEBUG] Слой {layer_idx}: топ-5 важных экспертов:")
                sorted_indices = torch.argsort(importances, descending=True)
                for rank in range(min(5, len(importances))):
                    idx = sorted_indices[rank].item()
                    print(f"    Expert {idx}: importance = {importances[idx]:.4f}")

    print(f"[DEBUG] Применение bias завершено для всех слоев")
    return model


def apply_expert_bias_v2(model, expert_importance, all_router_logits):
    print(f"[DEBUG] Применение смещения (bias) к экспертам с eps = std(|logits|) / num_experts")
    num_layers = len(model.model.layers)

    for layer_idx in range(num_layers):
        print(f"[DEBUG] Обработка слоя {layer_idx}")
        layer = model.model.layers[layer_idx]

        if len(expert_importance[layer_idx]) == 0:
            print(f"[DEBUG] Слой {layer_idx}: нет данных о важности экспертов, пропускаем")
            continue

        importances = torch.zeros(layer.mlp.num_experts)
        for expert_idx, imp in expert_importance[layer_idx].items():
            importances[expert_idx] = imp

        print(f"[DEBUG] Слой {layer_idx}: сырые важности - min={importances.min():.4f}, max={importances.max():.4f}, mean={importances.mean():.4f}")

        mean_imp = importances.mean()
        std_imp = importances.std()
        print(f"[DEBUG] Слой {layer_idx}: mean={mean_imp:.4f}, std={std_imp:.4f}")

        if std_imp > 1e-8:
            importances = (importances - mean_imp) / std_imp
            print(f"[DEBUG] Слой {layer_idx}: нормализованные важности - min={importances.min():.4f}, max={importances.max():.4f}")
        else:
            print(f"[DEBUG] Слой {layer_idx}: std слишком мал ({std_imp:.6f}), пропускаем нормализацию")

        print(f"[DEBUG] Вычисляем eps = std(|logits|) / num_experts")
        logit_means_per_sample = []
        for router_logits in tqdm(all_router_logits[layer_idx]):
            logit_means_per_sample.append(router_logits.abs().std().item())

        logit_means_per_dataset = np.mean(logit_means_per_sample)

        with torch.no_grad():
            if layer.mlp.gate.bias is None:
                num_experts = layer.mlp.num_experts
                layer.mlp.gate.bias = nn.Parameter(torch.zeros(num_experts, dtype=layer.mlp.gate.weight.dtype, device=layer.mlp.gate.weight.device))
                print(f"[DEBUG] Слой {layer_idx}: создан новый bias параметр (size={num_experts})")
            
            eps = logit_means_per_dataset / layer.mlp.num_experts
            print(f"[DEBUG] Слой {layer_idx}: eps = {eps}")

            old_bias = layer.mlp.gate.bias.clone()
            layer.mlp.gate.bias += eps * importances.to(layer.mlp.gate.bias.device)
            bias_change = torch.norm(layer.mlp.gate.bias - old_bias).item()
            print(f"[DEBUG] Слой {layer_idx}: bias обновлен (изменение нормы: {bias_change:.6f})")

        print(f"[DEBUG] Слой {layer_idx}: топ-5 важных экспертов:")
        sorted_indices = torch.argsort(importances, descending=True)
        for rank in range(min(5, len(importances))):
            idx = sorted_indices[rank].item()
            print(f"    Expert {idx}: importance = {importances[idx]:.4f}")

    print(f"[DEBUG] Применение bias завершено для всех слоев")
    return model


def apply_expert_bias_v3(model, expert_importance, all_router_logits):
    print(f"[DEBUG] Применение смещения (bias) к экспертам с eps = sqrt(mean(|logits|) * std(|logits|)) / num_experts")
    num_layers = len(model.model.layers)

    for layer_idx in range(num_layers):
        print(f"[DEBUG] Обработка слоя {layer_idx}")
        layer = model.model.layers[layer_idx]

        if len(expert_importance[layer_idx]) == 0:
            print(f"[DEBUG] Слой {layer_idx}: нет данных о важности экспертов, пропускаем")
            continue

        importances = torch.zeros(layer.mlp.num_experts)
        for expert_idx, imp in expert_importance[layer_idx].items():
            importances[expert_idx] = imp

        print(f"[DEBUG] Слой {layer_idx}: сырые важности - min={importances.min():.4f}, max={importances.max():.4f}, mean={importances.mean():.4f}")

        mean_imp = importances.mean()
        std_imp = importances.std()
        print(f"[DEBUG] Слой {layer_idx}: mean={mean_imp:.4f}, std={std_imp:.4f}")

        if std_imp > 1e-8:
            importances = (importances - mean_imp) / std_imp
            print(f"[DEBUG] Слой {layer_idx}: нормализованные важности - min={importances.min():.4f}, max={importances.max():.4f}")
        else:
            print(f"[DEBUG] Слой {layer_idx}: std слишком мал ({std_imp:.6f}), пропускаем нормализацию")

        print(f"[DEBUG] Вычисляем eps = sqrt(mean(|logits|) * std(|logits|)) / num_experts")
        logit_means_per_sample = []
        logit_stds_per_sample = []
        for router_logits in tqdm(all_router_logits[layer_idx]):
            logit_means_per_sample.append(router_logits.abs().mean().item())
            logit_stds_per_sample.append(router_logits.abs().std().item())

        logit_means_per_dataset = np.mean(logit_means_per_sample)
        logit_stds_per_dataset = np.mean(logit_stds_per_sample)

        with torch.no_grad():
            if layer.mlp.gate.bias is None:
                num_experts = layer.mlp.num_experts
                layer.mlp.gate.bias = nn.Parameter(torch.zeros(num_experts, dtype=layer.mlp.gate.weight.dtype, device=layer.mlp.gate.weight.device))
                print(f"[DEBUG] Слой {layer_idx}: создан новый bias параметр (size={num_experts})")
            
            geometric_mean = np.sqrt(logit_means_per_dataset * logit_stds_per_dataset)
            eps = geometric_mean / layer.mlp.num_experts
            print(f"[DEBUG] Слой {layer_idx}: eps = {eps}")

            old_bias = layer.mlp.gate.bias.clone()
            layer.mlp.gate.bias += eps * importances.to(layer.mlp.gate.bias.device)
            bias_change = torch.norm(layer.mlp.gate.bias - old_bias).item()
            print(f"[DEBUG] Слой {layer_idx}: bias обновлен (изменение нормы: {bias_change:.6f})")

        print(f"[DEBUG] Слой {layer_idx}: топ-5 важных экспертов:")
        sorted_indices = torch.argsort(importances, descending=True)
        for rank in range(min(5, len(importances))):
            idx = sorted_indices[rank].item()
            print(f"    Expert {idx}: importance = {importances[idx]:.4f}")

    print(f"[DEBUG] Применение bias завершено для всех слоев")
    return model


def apply_expert_bias_v4(model, expert_importance, all_router_logits, expert_outputs_mean=None):
    num_layers = len(model.model.layers)

    for layer_idx in range(num_layers):
        is_debug_layer = layer_idx in [1, 7, num_layers - 1]
        if is_debug_layer:
            print(f"[DEBUG] Обработка слоя {layer_idx}")
        
        layer = model.model.layers[layer_idx]
        num_experts = layer.mlp.num_experts
        top_k = 8

        if len(expert_importance[layer_idx]) == 0:
            if is_debug_layer:
                print(f"[DEBUG] Слой {layer_idx}: нет данных о важности экспертов, пропускаем")
            continue

        importances = torch.zeros(num_experts)
        for expert_idx, imp in expert_importance[layer_idx].items():
            importances[expert_idx] = imp

        if is_debug_layer:
            print(f"[DEBUG] Слой {layer_idx}: важности до нормализации: {' '.join([f'{x:.4f}' for x in importances.tolist()])}")

        mean_imp = importances.mean()
        importances = importances - mean_imp
        max_abs = importances.abs().max()
        if max_abs > 1e-8:
            importances = importances / max_abs
            if is_debug_layer:
                print(f"[DEBUG] Слой {layer_idx}: важности после Max-Abs нормализации: {' '.join([f'{x:.4f}' for x in importances.tolist()])}")
        
        all_logits_for_layer = torch.stack(all_router_logits[layer_idx])
        mean_router_logits = all_logits_for_layer.mean(dim=0)
        logit_std = mean_router_logits.std().item()

        if is_debug_layer:
            print(f"[DEBUG] Слой {layer_idx}: router logit (pre-softmax) mean: {' '.join([f'{x:.4f}' for x in mean_router_logits.tolist()])}")
            print(f"[DEBUG] Слой {layer_idx}: router logit (pre-softmax) global mean: {mean_router_logits.mean().item():.4f}, global std: {logit_std:.4f}")
            
            last_3_logits = all_router_logits[layer_idx][-3:]
            for i, token_logits in enumerate(last_3_logits):
                print(f"[DEBUG] Слой {layer_idx}: router logit (pre-softmax) token -{3-i}: {' '.join([f'{x:.4f}' for x in token_logits.flatten().tolist()])}")

        with torch.no_grad():
            if layer.mlp.gate.bias is None:
                layer.mlp.gate.bias = nn.Parameter(torch.zeros(num_experts, dtype=layer.mlp.gate.weight.dtype, device=layer.mlp.gate.weight.device))
            
            eps = (logit_std) / (num_experts / top_k)
            
            old_bias = layer.mlp.gate.bias.clone()
            layer.mlp.gate.bias += eps * importances.to(layer.mlp.gate.bias.device)
            
            if is_debug_layer:
                print(f"[DEBUG] Слой {layer_idx}: eps = {eps:.6f} (S={SENSITIVITY}, std={logit_std:.4f}, N/K={num_experts/top_k})")
                bias_change = torch.norm(layer.mlp.gate.bias - old_bias).item()
                print(f"[DEBUG] Слой {layer_idx}: bias обновлен (изменение нормы: {bias_change:.6f})")
                print(f"[DEBUG] Слой {layer_idx}: Итоговый bias: {' '.join([f'{x:.4f}' for x in layer.mlp.gate.bias.tolist()])}")

                for exp_idx in range(len(layer.mlp.experts)):
                    if expert_outputs_mean and exp_idx in expert_outputs_mean[layer_idx]:
                        act_vals = expert_outputs_mean[layer_idx][exp_idx].flatten()[:20].tolist()
                        print(f"[DEBUG] Expert {exp_idx} mean output activations (h_i): {' '.join([f'{x:.4f}' for x in act_vals])}")

    print(f"[DEBUG] Применение bias завершено для всех слоев")
    return model


def apply_expert_bias_v5(model, expert_importance, all_router_logits, expert_outputs_mean=None):
    num_layers = len(model.model.layers)

    for layer_idx in range(num_layers):
        is_debug_layer = layer_idx in [1, 7, num_layers - 1]
        if is_debug_layer:
            print(f"[DEBUG] Обработка слоя {layer_idx}")
        
        layer = model.model.layers[layer_idx]
        num_experts = layer.mlp.num_experts
        top_k = 4

        if len(expert_importance[layer_idx]) == 0:
            if is_debug_layer:
                print(f"[DEBUG] Слой {layer_idx}: нет данных о важности экспертов, пропускаем")
            continue

        importances = torch.zeros(num_experts)
        for expert_idx, imp in expert_importance[layer_idx].items():
            importances[expert_idx] = imp

        mean_imp = importances.mean()
        importances = importances - mean_imp
        max_abs = importances.abs().max()
        if max_abs > 1e-8:
            importances = importances / max_abs
        
        all_logits_for_layer = torch.stack(all_router_logits[layer_idx])
        mean_router_logits = all_logits_for_layer.mean(dim=0)
        
        sorted_logits, _ = torch.sort(mean_router_logits, descending=True)
        logit_gap = (sorted_logits[0] - sorted_logits[top_k]).item()
        eps = max(logit_gap, 1e-4)

        if is_debug_layer:
            print(f"[DEBUG] Слой {layer_idx}: важности до нормализации: {' '.join([f'{x:.4f}' for x in (importances * max_abs + mean_imp).tolist()])}")
            print(f"[DEBUG] Слой {layer_idx}: важности после Max-Abs: {' '.join([f'{x:.4f}' for x in importances.tolist()])}")
            print(f"[DEBUG] Слой {layer_idx}: router logit (pre-softmax) mean: {' '.join([f'{x:.4f}' for x in mean_router_logits.tolist()])}")
            print(f"[DEBUG] Слой {layer_idx}: top-1 logit: {sorted_logits[0]:.4f}, top-{top_k+1} logit: {sorted_logits[top_k]:.4f}")
            print(f"[DEBUG] Слой {layer_idx}: eps (logit gap) = {eps:.6f}")

        with torch.no_grad():
            if layer.mlp.gate.bias is None:
                layer.mlp.gate.bias = nn.Parameter(torch.zeros(num_experts, dtype=layer.mlp.gate.weight.dtype, device=layer.mlp.gate.weight.device))
            
            old_bias = layer.mlp.gate.bias.clone()
            layer.mlp.gate.bias += eps * importances.to(layer.mlp.gate.bias.device)
            
            if is_debug_layer:
                bias_change = torch.norm(layer.mlp.gate.bias - old_bias).item()
                print(f"[DEBUG] Слой {layer_idx}: bias обновлен (изменение нормы: {bias_change:.6f})")
                print(f"[DEBUG] Слой {layer_idx}: Итоговый bias: {' '.join([f'{x:.4f}' for x in layer.mlp.gate.bias.tolist()])}")

    print(f"[DEBUG] Применение bias завершено для всех слоев")
    return model


def apply_expert_bias_v6(model, expert_importance, all_router_logits, expert_outputs_mean=None):
    num_layers = len(model.model.layers)

    for layer_idx in range(num_layers):
        is_debug_layer = layer_idx in [1, 7, num_layers - 1]
        if is_debug_layer:
            print(f"[DEBUG] Обработка слоя {layer_idx}")
        
        layer = model.model.layers[layer_idx]
        num_experts = layer.mlp.num_experts

        if len(expert_importance[layer_idx]) == 0:
            if is_debug_layer:
                print(f"[DEBUG] Слой {layer_idx}: нет данных о важности экспертов, пропускаем")
            continue

        importances = torch.zeros(num_experts)
        for expert_idx, imp in expert_importance[layer_idx].items():
            importances[expert_idx] = imp

        mean_imp = importances.mean()
        importances = importances - mean_imp
        max_abs = importances.abs().max()
        if max_abs > 1e-8:
            importances = importances / max_abs
        
        all_logits_tensor = torch.stack(all_router_logits[layer_idx]) # [num_tokens, num_experts]
        global_std = all_logits_tensor.std().item()
        
        eps = (4.0 * global_std) / num_experts

        if is_debug_layer:
            print(f"[DEBUG] Слой {layer_idx}: важности до нормализации: {' '.join([f'{x:.4f}' for x in (importances * max_abs + mean_imp).tolist()])}")
            print(f"[DEBUG] Слой {layer_idx}: важности после Max-Abs: {' '.join([f'{x:.4f}' for x in importances.tolist()])}")
            print(f"[DEBUG] Слой {layer_idx}: global logit std: {global_std:.4f}")
            print(f"[DEBUG] Слой {layer_idx}: eps (4-sigma / N) = {eps:.6f}")

        with torch.no_grad():
            if layer.mlp.gate.bias is None:
                layer.mlp.gate.bias = nn.Parameter(torch.zeros(num_experts, dtype=layer.mlp.gate.weight.dtype, device=layer.mlp.gate.weight.device))
            
            old_bias = layer.mlp.gate.bias.clone()
            layer.mlp.gate.bias += eps * importances.to(layer.mlp.gate.bias.device)
            
            if is_debug_layer:
                bias_change = torch.norm(layer.mlp.gate.bias - old_bias).item()
                print(f"[DEBUG] Слой {layer_idx}: bias обновлен (изменение нормы: {bias_change:.6f})")
                print(f"[DEBUG] Слой {layer_idx}: Итоговый bias: {' '.join([f'{x:.4f}' for x in layer.mlp.gate.bias.tolist()])}")

    print(f"[DEBUG] Применение bias завершено для всех слоев")
    return model


if __name__ == "__main__":
    print("[DEBUG] ===== НАЧАЛО РАБОТЫ =====")

    print(f"[DEBUG] Проверка CUDA: доступно = {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[DEBUG] Количество GPU: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"[DEBUG] GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"[DEBUG] Память GPU {i}: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")

    print("[DEBUG] Загрузка токенизатора...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-MoE-A2.7B")
    print("[DEBUG] Токенизатор загружен")

    print("[DEBUG] Загрузка модели...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen1.5-MoE-A2.7B",
        torch_dtype=torch.float16,
        device_map="auto",
        num_experts_per_tok=8
    )


    print(f"[DEBUG] Модель загружена на устройство: {model.device}")
    print(f"[DEBUG] Тип модели: {model.dtype}")

    '''
    print("\n[DEBUG] ===== ТЕСТИРОВАНИЕ ДО ОБУЧЕНИЯ =====")
    accuracy = test_accuracy(model, tokenizer, test_samples_math)
    print("BEFORE college_mathematics:", accuracy)

    accuracy = test_accuracy(model, tokenizer, test_samples_alg)
    print("BEFORE abstract_algebra:", accuracy)

    accuracy = test_accuracy(model, tokenizer, test_samples_all)
    print("BEFORE all:", accuracy)
    print()
    '''

    print("\n[DEBUG] ===== ВЫЧИСЛЕНИЕ ВАЖНОСТИ ЭКСПЕРТОВ =====")
    expert_importance, mlp_inputs, expert_outputs_mean = compute_expert_importance(model, tokenizer, test_samples_math)

    print("\n[DEBUG] ===== ПРИМЕНЕНИЕ СМЕЩЕНИЯ К ЭКСПЕРТАМ =====")
    model = apply_expert_bias_v6(model, expert_importance, mlp_inputs, expert_outputs_mean)

    print("\n[DEBUG] ===== ТЕСТИРОВАНИЕ ПОСЛЕ ОБУЧЕНИЯ =====")
    acc_math = test_accuracy(model, tokenizer, test_samples_math)
    print("AFTER college_mathematics:", acc_math)

    acc_alg = test_accuracy(model, tokenizer, test_samples_alg)
    print("AFTER abstract_algebra:", acc_alg)

    acc_all = test_accuracy(model, tokenizer, test_samples_all)
    print("AFTER all:", acc_all)

    # txt_filename = "WES_metrics.txt"
    # with open(txt_filename, "a", encoding="utf-8") as f:
    #     f.write(f"\n{os.path.basename(__file__)} top-k all:\n")
    #     f.write(f"Math Accuracy:    {acc_math:.4f}\n")
    #     f.write(f"Algebra Accuracy: {acc_alg:.4f}\n")
    #     f.write(f"Overall Accuracy: {acc_all:.4f}\n")
    #     f.write("-" * 30 + "\n")
    # print(f"Метрики добавлены в {txt_filename}")

    print("[DEBUG] ===== РАБОТА ЗАВЕРШЕНА =====")