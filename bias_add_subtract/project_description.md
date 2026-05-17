# Project description

## 1. High-level summary

Проект `WES` - это исследовательский Python-скрипт для анализа и модификации routing logic в MoE-модели `qwen-moe-chat`. Весь рабочий pipeline находится в `main.py`; дополнительные файлы `CODEX.md` и `EXPERT_IMPORTANCE.md` описывают текущее состояние экспериментов и формулу expert importance.

Основная задача проекта: без дообучения модели найти экспертов и слои, которые локально помогают или мешают правильному ответу на MMLU-задачи, и затем изменить router logits через sparse bias для выбранных экспертов. После этого модель повторно оценивается на MMLU-подобных доменных датасетах.

Проект работает с задачами множественного выбора `A/B/C/D`. Для calibration модель генерирует короткий ответ, находится первый generation step, на котором можно извлечь букву ответа, затем на этой позиции считается loss по четырём logits `A/B/C/D`. По градиентам этого loss оценивается signed importance экспертов, после чего важность конвертируется в bias к `layer.mlp.gate.bias`.

## 2. Research goal

Исследовательская цель проекта: проверить, можно ли управлять поведением MoE-модели через малое runtime-вмешательство в router, не меняя веса экспертов и не дообучая модель.

Проверяемая гипотеза:

- на calibration dataset можно определить экспертов, которые положительно или отрицательно влияют на вероятность правильного ответа;
- не все слои одинаково полезны для вмешательства;
- если выбрать только high-impact слои и high-impact экспертов, то можно немного улучшить качество на домене или снизить регрессии относительно baseline;
- gap между router logits важен, потому что bias должен быть соразмерен устойчивости текущего top-k routing.

Практически проект исследует domain-specific routing biasing: calibration берётся из одного домена MMLU, а evaluation выполняется на этом и близких доменных датасетах.

## 3. Core idea

Модель загружается из локального пути:

```text
/mnt/storage00/usr/ishukin/models/qwen-moe-chat
```

По коду это `AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.float16, device_map="auto")` и `AutoTokenizer.from_pretrained(..., trust_remote_code=True)`.

В модели предполагается Qwen MoE-архитектура, где каждый transformer layer содержит `layer.mlp`, а внутри MLP есть:

- `layer.mlp.gate` - router/gating linear layer;
- `layer.mlp.experts` - список expert modules;
- `layer.mlp.top_k` - сколько экспертов выбирается на токен;
- `layer.mlp.num_experts` - число экспертов в слое;
- `layer.mlp.norm_topk_prob` - нормировать ли top-k routing weights.

Что изменяется:

- веса модели и веса экспертов не дообучаются;
- в runtime создаётся или обновляется `layer.mlp.gate.bias`;
- bias добавляется к router logits до выбора top-k экспертов;
- положительные эксперты получают положительный delta bias;
- отрицательные эксперты получают отрицательный delta bias, потому что их normalized importance отрицательная;
- после добавления bias router может выбрать другой top-k набор экспертов для следующих forward/generation calls.

Почему выбираются top-k эксперты:

- MoE routing активирует не всех экспертов, а только top-k по router probabilities/logits;
- importance считается только для фактически выбранных экспертов или расширенного набора top-k/top-k*2/all, в зависимости от `loss_top_k`;
- bias имеет смысл применять разреженно: к тем экспертам, которые являются кандидатами на изменение routing.

Почему анализируются positive и negative importance:

- positive importance означает, что вклад эксперта локально связан со снижением loss правильного ответа;
- negative importance означает, что вклад эксперта локально связан с ростом loss;
- positive experts кандидаты на promotion через router bias;
- negative experts кандидаты на suppression через отрицательный bias.

Почему выбираются отдельные слои:

- слои различаются по силе expert importance и по устойчивости router decision;
- если bias применять ко всем слоям, можно получить шумное и слишком широкое вмешательство;
- поэтому в проекте проверяются fixed layer selection, threshold-based selection, quantile selection и automatic Otsu selection.

Зачем calibration datasets:

- calibration dataset даёт небольшой набор задач, на которых собираются градиенты, router logits и expert importance;
- текущий экспериментальный протокол использует 100 calibration samples для домена, если calibration dataset является `*_part`;
- затем evaluation выполняется на полном оригинальном размере тестовых датасетов.

Зачем сравнение на MMLU-подобных датасетах:

- MMLU содержит много предметных доменов в одинаковом формате `A/B/C/D`;
- это позволяет калибровать на одном домене и проверять перенос на близкие предметы;
- `all_part` используется как дополнительная sanity-check выборка для math-режима.

## 4. Project structure

| Path                                                | Type                 | Purpose                                                                                                                                                                                                                                                                                                                                                                                |
| --------------------------------------------------- | -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `main.py`                                         | script               | Главный и фактически единственный executable pipeline: загрузка MMLU, prompt formatting, generation, answer parsing, accuracy evaluation, expert importance, layer/expert selection, router bias application, CLI.                                                                                                                               |
| `CODEX.md`                                        | documentation        | Техническая карта текущего состояния проекта. Описывает функции, конфиги, домены, CLI и текущий активный метод `v6`. Использована как вспомогательный источник, но приоритет в этом документе отдаётся `main.py`. |
| `EXPERT_IMPORTANCE.md`                            | documentation        | Формально описывает формулу expert importance `- dot(g, w_i * h_i)` и интуицию first-order/Taylor-style оценки.                                                                                                                                                                                                                              |
| `project_description.md`                          | documentation        | Этот файл. Сводное reverse-engineering описание проекта, логики и экспериментов.                                                                                                                                                                                                                                                     |
| `/home/anmilka/MoE/mmlu_data`                     | data/cache, external | Жёстко заданный путь cache для MMLU datasets. Не находится в репозитории.`main.py` либо читает `load_from_disk`, либо скачивает `cais/mmlu` и сохраняет туда датасеты.                                                                                                                  |
| `/mnt/storage00/usr/ishukin/models/qwen-moe-chat` | model, external      | Жёстко заданный локальный путь модели. Не находится в репозитории.                                                                                                                                                                                                                                                             |

### Key code areas in `main.py`

| Code area             | Functions / variables                                                                                                                                                                                              | Responsibility                                                                                                                                      |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| Global config         | `SEED`, `BIAS_START_LAYER`, `POSITIVE_BIAS_LAYERS`, `NEGATIVE_BIAS_LAYERS`, `NUM_EXP_POS_BIAS_APPLY`, `NUM_EXP_NEG_BIAS_APPLY`, `POSITIVE_TOP3_MEAN_THRESHOLD`, `NEGATIVE_TOP3_ABS_MEAN_THRESHOLD` | Управление random seed, layer gating и историческими/ручными настройками bias selection.                  |
| Dataset cache/loading | `data_folder`, `saved_paths`, `load_from_disk`, `load_dataset("cais/mmlu", ...)`                                                                                                                           | Загрузка и кэширование MMLU subsets.                                                                                            |
| Prompting             | `pretty_subject`, `format_mmlu_question`, `build_chat_texts`                                                                                                                                                 | Преобразование MMLU example в chat prompt.                                                                                           |
| Answer extraction     | `first_option_postprocess`                                                                                                                                                                                       | Regex-based извлечение первой валидной буквы `A/B/C/D` из generated text.                                          |
| Letter logits         | `get_letter_token_ids`, `build_abcd_logits`, `choose_preferred_letter_token_id`                                                                                                                              | Поиск token ids для букв и сбор 4 logits `A/B/C/D` из full vocab logits.                                                       |
| Evaluation            | `generate_responses`, `test_accuracy`, `evaluate_named_datasets`                                                                                                                                             | Batched generation, extraction и accuracy.                                                                                                         |
| Expert importance     | `compute_expert_importance`                                                                                                                                                                                      | Calibration loop: generation до 20 токенов, CE на answer step, hooks, gradients, router logits, expert importance.                       |
| Bias methods          | `apply_expert_bias`, `apply_expert_bias_v4`, `apply_expert_bias_v5`, `apply_expert_bias_v6`, `apply_expert_bias_v7`, `apply_expert_bias_default`                                                       | Разные версии router bias modification. Активная экспериментальная ветка по умолчанию -`v6`. |
| Layer thresholding    | `otsu_threshold`                                                                                                                                                                                                 | Otsu threshold для автоматического выбора слоёв по layer scores.                                                     |
| Domain setup          | `get_domain_configs`                                                                                                                                                                                             | Связывает calibration dataset и evaluation datasets для `math`, `physics`, `history`, `biology`, `chemistry`.                |
| CLI / entrypoint      | `if __name__ == "__main__"`                                                                                                                                                                                      | CLI args, режимы `single`/`all`, baseline branch, calibration, modification, evaluation.                                                  |

### Entrypoints

Главный entrypoint:

```bash
python main.py
```

Фактическое поведение по умолчанию:

```bash
python main.py --method v6 --mode single --domain math --loss_top_k top_k
```

## 5. Main execution flow

Фактический pipeline в текущем `main.py`:

1. На уровне import/module execution выставляются seeds и загружаются или скачиваются MMLU datasets.
2. CLI парсит `--method`, `--mode`, `--domain`, `--loss_top_k`, `--eps_value`.
3. `get_domain_configs()` выбирает calibration dataset и eval datasets для домена.
4. Если `--mode all` и method не `baseline`, текущий процесс запускает отдельный subprocess для каждого домена: `math`, `physics`, `history`, `biology`, `chemistry`.
5. `load_model_and_tokenizer()` загружает tokenizer и model.
6. Если `--method baseline`, выполняется текущий baseline block. В коде большинство baseline datasets закомментированы; активны только `high_school_chemistry` и `college_chemistry`.
7. Для выбранного домена вызывается `compute_expert_importance(model, tokenizer, calibration_dataset, args.loss_top_k)`.
8. Внутри calibration для каждого example:
   - строится MMLU prompt;
   - применяется chat template;
   - генерируется до 20 новых токенов;
   - ищется первый prefix generation, где regex извлекает `A/B/C/D`;
   - выполняется forward на `prompt + generated_prefix_before_answer`;
   - берутся logits на позиции, которая предсказывает answer token;
   - из full vocabulary logits собираются logits для `A/B/C/D`;
   - считается CE loss относительно gold answer;
   - через hooks собираются MLP inputs и gradients;
   - пересчитывается router routing на той же позиции;
   - считается expert importance;
   - сохраняются router logits для слоя.
9. `run_calibration_method()` применяет выбранный bias method.
10. Для `v6` строятся per-layer stats, выбираются positive/negative layers, выбираются experts внутри layer, затем обновляется `layer.mlp.gate.bias`.
11. После bias application вызывается `evaluate_named_datasets()` для eval datasets домена.
12. Для `math` дополнительно оценивается `all_part`.
13. Результаты не сохраняются в файл: они печатаются в stdout.

Pipeline в виде схемы:

```text
1. Загрузка модели.
2. Загрузка tokenizer / processor.
3. Загрузка датасета.
4. Подготовка prompt-ов.
5. Генерация / forward pass.
6. Извлечение ответа A/B/C/D.
7. Расчёт logits для A/B/C/D.
8. Расчёт CE / loss.
9. Сбор router logits / expert scores / importance.
10. Выбор слоёв.
11. Выбор экспертов.
12. Применение bias / modification.
13. Повторная evaluation.
14. Печать результатов в stdout.
```

Отличие от указанного в задании generic pipeline: текущий код не имеет отдельного output path и не сохраняет результаты автоматически.

## 6. Model modification logic

Вмешательство реализовано как изменение router bias:

```python
layer.mlp.gate.bias += delta_bias
```

или для `default`:

```python
layer.mlp.gate.bias.data = eps * importances
```

Если `layer.mlp.gate.bias is None`, код создаёт новый `nn.Parameter` нулевого размера `num_experts`.

Модель меняется только в runtime текущего Python-процесса. Файлы весов модели не перезаписываются, checkpoint не сохраняется. После завершения процесса изменения исчезают.

Механически bias влияет на router logits. Router затем делает softmax/top-k selection уже с учётом смещённых logits. Поэтому:

- positive delta повышает шанс эксперта попасть в top-k;
- negative delta снижает шанс эксперта попасть в top-k;
- если исходный gap между top-k и top-k+1 большим, слабый bias может ничего не изменить;
- если gap маленький, даже небольшой bias может поменять routing.

### Bias methods

| Method       | Function                                           | Current status                     | Logic                                                                                                                                                                                                                                                                                                                                                                       |
| ------------ | -------------------------------------------------- | ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `baseline` | `run_calibration_method` returns model unchanged | Active CLI option                  | Без bias modification. Важно: текущий baseline branch всё равно затем доходит до `compute_expert_importance`, потому что после baseline block нет `SystemExit`; это выглядит как неочевидное поведение и требует проверки перед запуском baseline. |
| `original` | `apply_expert_bias`                              | Legacy/current option              | Z-score normalization по expert importances,`eps = top_k * mean(std(abs(router_logits))) / num_experts`, bias применяется к слоям `layer_idx < BIAS_START_LAYER`. Печатает top-5 и bottom-5 experts.                                                                                                                                        |
| `v4`       | `apply_expert_bias_v4`                           | Legacy/current option              | Max-Abs normalization,`eps = (logit_std * SENSITIVITY) / (num_experts / top_k)`, layer gate использует `layer_idx < BIAS_START_LAYER: continue`, то есть обрабатываются слои `>= BIAS_START_LAYER`.                                                                                                                                 |
| `v5`       | `apply_expert_bias_v5`                           | Legacy/current option              | Max-Abs normalization,`eps = max(logit_gap, 1e-4)`, где `logit_gap = sorted_logits[0] - sorted_logits[top_k]` по mean router logits.                                                                                                                                                                                                                               |
| `v6`       | `apply_expert_bias_v6`                           | Default and main experimental path | Gap-normalized sparse bias. Выбирает experts адаптивно, выбирает layers через Otsu/double Otsu или manual allowlists, применяет sparse `delta_bias = eps * sparse_importances`.                                                                                                                                                 |
| `v7`       | `apply_expert_bias_v7`                           | Legacy/current option              | Max-Abs normalization,`eps = IQR(router_logits) / num_experts`, где IQR = q75 - q25.                                                                                                                                                                                                                                                                                   |
| `default`  | `apply_expert_bias_default`                      | Sweep/fixed eps option             | Max-Abs normalization, fixed `eps_value`. В режиме `--method default --loss_top_k top_k` запускается sweep по eps values.                                                                                                                                                                                                                           |

## 7. Calibration logic

Calibration реализована в `compute_expert_importance`.

### Calibration datasets

`get_domain_configs()` задаёт доменные calibration datasets:

| Domain        | Calibration dataset in code                     | Evaluation datasets                                                                                                                                            |
| ------------- | ----------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `math`      | `test_samples_math` (`college_mathematics`) | `college_mathematics`, `abstract_algebra`, `elementary_mathematics`, `high_school_mathematics`, плюс `all_part` только в math single-mode |
| `physics`   | `test_samples_college_physics`                | `college_physics`, `conceptual_physics`, `high_school_physics`                                                                                           |
| `history`   | `test_samples_high_school_world_history_part` | `high_school_world_history`, `high_school_european_history`, `high_school_us_history`, `prehistory`                                                    |
| `biology`   | `test_samples_college_biology_part`           | `college_biology`, `high_school_biology`                                                                                                                   |
| `chemistry` | `test_samples_college_chemistry_part`         | `college_chemistry`, `high_school_chemistry`                                                                                                               |

По экспериментальному протоколу из предоставленной таблицы:

- `*` рядом с датасетом означает, что этот датасет использовался для калибровки;
- размер calibration dataset всегда 100 samples;
- для тестирования качества всегда использовался оригинальный полный размер датасета;
- например, если `high_school_world_history = 268 samples`, для calibration берутся random 100 samples, а evaluation идёт на всех 268 samples.

В текущем коде это полностью явно реализовано для `*_part` датасетов через `.shuffle(seed=SEED).select(range(100))`. Для `college_physics` код тоже делает `.select(range(100))` прямо в `test_samples_college_physics`, поэтому этот объект используется и как calibration, и как evaluation в текущем коде. Для `math` текущий код использует `test_samples_math` без `.select(range(100))`; это расходится с протоколом таблицы и требует ручной проверки, если нужно воспроизводить старые эксперименты строго по 100 calibration samples.

### How samples are selected

При отсутствии cache:

```python
mmlu_dataset["test"].shuffle(seed=SEED)
```

Для 100-sample subsets:

```python
mmlu_dataset["test"].shuffle(seed=SEED).select(range(100))
```

Seed фиксирован:

```python
SEED = 1235
```

### Data collected during calibration

Для каждого calibration example собирается:

- generated text;
- extracted answer generation step;
- CE loss на позиции answer token;
- MLP input для каждого слоя через forward hook;
- gradient по выходу MLP для каждого слоя через backward hook;
- router logits для выбранной token position;
- per-layer/per-expert accumulated importance.

Возвращаемые объекты:

```python
expert_importance, all_router_logits
```

где:

- `expert_importance[layer_idx][expert_idx] += importance`;
- `all_router_logits[layer_idx].append(router_logits[grad_token_idx])`.

## 8. Answer extraction and CE/loss computation

### Prompt format

`format_mmlu_question(example)` строит prompt:

```text
There is a single choice question about {subject}. Answer the question by replying only with letter A, B, C or D.
Question: ...
A. ...
B. ...
C. ...
D. ...
Answer:
```

Затем `build_chat_texts()` оборачивает prompt в chat template tokenizer:

```python
tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```

### Answer extraction

`first_option_postprocess(text, options='ABCD', cushion=True)` ищет первую валидную букву через набор regex patterns. В конце добавляются cushion patterns:

```text
([{options}]):
([{options}])
```

Это делает extraction довольно permissive: если в тексте есть одиночная буква `A/B/C/D`, она может быть извлечена.

### CE / loss algorithm

Текущий алгоритм в `compute_expert_importance`:

```text
Как считается CE / loss по шагам:
1. Генерируется до 20 токенов.
2. Находится первый шаг, где извлеклась буква ответа.
3. На этой позиции берутся logits словаря.
4. Из них извлекаются 4 логита: A / B / C / D.
5. Над этими четырьмя логитами считается Cross Entropy относительно gold-ответа.
```

Техническая детализация:

1. `model.generate(..., max_new_tokens=generation_max_new_tokens, do_sample=False)` генерирует continuation. По умолчанию `generation_max_new_tokens=20`.
2. Код декодирует prefix generation для каждого `step`:

   ```python
   prefix_text = tokenizer.decode(generated_ids[0, : step + 1], skip_special_tokens=False)
   prefix_answer = first_option_postprocess(prefix_text.strip())
   ```
3. Первый `step`, где `prefix_answer in ["A", "B", "C", "D"]`, сохраняется как `extracted_gen_idx`.
4. Conditioning sequence строится как `prompt + generated_prefix_before_answer`. Если answer letter была первым generated token, conditioning sequence равна prompt.
5. `pred_pos = conditioning_ids.shape[1] - 1`. Это позиция logits, которые предсказывают следующий token, то есть extracted answer token.
6. Выполняется forward:

   ```python
   outputs = model(input_ids=conditioning_ids, output_hidden_states=True)
   logits_pos = outputs.logits[0, pred_pos, :]
   ```
7. `build_abcd_logits(logits_pos, letter_token_ids)` выбирает logits для букв `A/B/C/D`. Для каждой буквы берутся single-token variants `" A"` и `"A"`, если tokenizer кодирует их одним токеном. Если есть несколько token ids, используется `torch.max(logits_pos[ids])`.
8. Текущий loss:

   ```python
   target = torch.tensor([correct_idx], device=abcd_logits.device)
   loss = F.cross_entropy(abcd_logits.float().unsqueeze(0), target)
   ```
9. Для debug дополнительно считается:

   ```python
   probs_abcd = F.softmax(abcd_logits.float(), dim=-1)
   p_correct = probs_abcd[correct_idx].clamp_min(1e-12)
   ```

### Old loss vs new loss

В коде сохранён старый loss как комментарий:

```python
p_correct = probs_abcd[correct_idx].clamp_min(1e-12)
loss_correct = -torch.log(p_correct)
loss_wrong = 1.0 - p_correct
beta = (1.0 - p_correct).detach()
loss = loss_correct + beta * loss_wrong
```

Текущий активный loss - plain multiclass cross-entropy over A/B/C/D logits. Поэтому эксперименты с пометкой `new loss` относятся к текущей реализации после замены old mixed-scale loss на CE.

Также есть commented alternative для full-vocabulary softmax: сначала считать softmax по всему словарю, потом брать probability для token ids букв. Эта ветка не активна.

## 9. Expert importance computation

Expert importance считается в `compute_expert_importance` и дополнительно описана в `EXPERT_IMPORTANCE.md`.

Формула:

```text
importance_{l,e} = - < g, w_e * h_e >
```

где:

- `l` - layer index;
- `e` - expert index;
- `g` - gradient loss по выходу MLP на выбранной token position;
- `h_e` - output expert `e` на MLP input выбранной token position;
- `w_e` - routing weight expert `e` на этой token position;
- `w_e * h_e` - вклад эксперта в MLP mixture output.

В коде:

```python
importance = -torch.dot(g_token, (w_i * h_i).squeeze(0)).item()
expert_importance[layer_idx][expert_idx] += importance
```

Интерпретация:

- `torch.dot(g, contribution)` - first-order оценка того, как вклад эксперта влияет на loss;
- минус перед dot делает положительные значения "полезными": positive importance означает, что эксперт локально помогает снижать loss;
- negative importance означает, что эксперт локально толкает prediction в сторону увеличения loss.

### Routing set controlled by `loss_top_k`

Внутри importance computation код пересчитывает router weights:

```python
router_logits = layer.mlp.gate(hidden_flat)
routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
```

Затем выбирает effective top-k:

| `loss_top_k` | Effective experts used for importance |
| -------------- | ------------------------------------- |
| `top_k`      | `layer.mlp.top_k`                   |
| `top_k_x2`   | `layer.mlp.top_k * 2`               |
| `all`        | `layer.mlp.num_experts`             |

Если `layer.mlp.norm_topk_prob` включён, выбранные routing weights нормируются по выбранному набору.

### Positive vs negative importance

После накопления по calibration dataset importances нормализуются внутри bias methods.

В `v6`:

```python
importances = importances - importances.mean()
importances = importances / max_abs
```

Дальше:

- positive experts: `importances > 0`;
- negative experts: `importances < 0`;
- selected positive experts: importance выше среднего positive importance слоя;
- selected negative experts: absolute negative importance выше `mean(abs(negative)) + std(abs(negative))`.

### `eps`, `gap`, `mean(top3)`, `mean_gap(topk - top(k+1))`

В legacy experiments и debug logs встречаются несколько понятий:

- `eps` - scale factor, которым normalized importance переводится в router bias delta;
- `gap` - средний разрыв router logits между последним выбранным top-k экспертом и следующим кандидатом;
- `mean(top3)` - средняя importance top-3 selected positive experts в старой фиксированной логике;
- `mean_gap(topk - top(k+1))` - среднее по calibration examples значение `sorted_logits[:, top_k - 1] - sorted_logits[:, top_k]`;
- `mean_gap(top1 - top2)` - средний разрыв между лучшим и вторым router logit.

В `v6`:

```python
g1 = mean_gap(top1 - top2)
gk = mean_gap(topk - top(k+1))
eps = (g1 * gk) / max(g1 + gk, 1e-8)
safe_gap = max(gk, 1e-8)
```

Layer score:

```text
layer_score = (eps * mean(selected_importance)) / gap
```

В коде отдельно для signs:

```python
pos_auto_score = (eps * pos_mean) / safe_gap
neg_auto_score = (eps * neg_mean_abs) / safe_gap
```

Почему gap важен:

- router top-k зависит от относительного порядка logits;
- если `gap` большой, bias должен быть достаточно сильным, чтобы реально изменить routing;
- если `gap` маленький, слой чувствителен к малым bias updates;
- деление на `gap` повышает score слоёв, где selected importance может быть реализована меньшим вмешательством.

Layer score используется для автоматического выбора слоёв через Otsu threshold.

## 10. Layer selection strategies

В проекте есть несколько исторических и текущих стратегий. Не все они одновременно активны в текущем `main.py`; часть восстановлена по комментариям и таблице экспериментов.

| Strategy                                           | Где реализована / источник                                                                                                                                                                                                                                             | Parameters                                                                         | Что делает                                                                                                                                                                                         | Плюсы                                                                                                                    | Минусы / caveats                                                                                                                                                                 | Эксперименты                                                                                                |
| -------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Fixed first N layers                               | Исторический эксперимент `First 4 layers only`; в текущем коде прямой CLI-опции нет. Ближайший механизм - ручные `POSITIVE_BIAS_LAYERS`/`NEGATIVE_BIAS_LAYERS` или изменение `BIAS_START_LAYER`. | `N`, manual layer lists                                                          | Применяет bias только к первым N слоям.                                                                                                                                          | Простая baseline-гипотеза: ранние слои могут быть более общими.                  | В текущем коде не оформлено как параметр CLI; нужно менять код.                                                                        | `First 4 layers only`, `top-3+4layers`.                                                                             |
| Top-N layers by expert importance                  | Исторические эксперименты из таблицы.                                                                                                                                                                                                                       | `N=3`, `N=5`, metric по expert importance                                    | Выбирает слои с крупнейшей aggregated expert importance.                                                                                                                             | Проверяет, достаточно ли концентрироваться на самых "сильных" слоях. | Текущая реализация не содержит отдельной функции top-N layer selection; вероятно была старой версией кода.     | `top-3+3 biggest exp imp layers`, `top-3+5 biggest exp imp layers`.                                                 |
| Mean importance threshold                          | Исторический эксперимент; текущий код содержит constants `POSITIVE_TOP3_MEAN_THRESHOLD=0.6`, `NEGATIVE_TOP3_ABS_MEAN_THRESHOLD=0.6`, но они не используются активной `v6`.                                         | threshold, например `0.6`                                                | Выбирает слои, где mean selected importance выше threshold.                                                                                                                              | Интерпретируемый ручной порог.                                                                     | Не адаптируется под домен/распределение; в текущем коде не активен.                                                            | `top-3+mean imp > 0.6`, `new loss + layer select by mean imp > 0.6 + top-3 experts`.                                |
| Positive and negative thresholds separately        | Исторический эксперимент и comments.                                                                                                                                                                                                                                 | `pos_imp>0.6`, `neg_imp>0.7`                                                   | Отдельно выбирает positive и negative layers по разным thresholds.                                                                                                                 | Учитывает асимметрию полезных и вредных экспертов.                                | Нужно вручную подбирать thresholds.                                                                                                                               | `top-3+pos_imp>0.6,neg_imp>0.7`.                                                                                      |
| Quantile thresholds                                | Комментарии в `apply_expert_bias_v6`, experiment table.                                                                                                                                                                                                                        | `pos_q`, `neg_q`, например `0.60/0.90`, `0.70/0.80`, `0.50/0.90` | Выбирает layers выше quantile по layer score.                                                                                                                                                 | Более адаптивно, чем абсолютный threshold.                                                         | В текущем коде quantile block закомментирован; переменные `POSITIVE_BIAS_AUTO_QUANTILE`/`NEGATIVE_BIAS_AUTO_QUANTILE` отсутствуют. | `(eps * mean(top3)) / gap, pos_q=... neg_q=...`, `new loss + ... pos_q=0.60 neg_q=0.90`.                            |
| Otsu threshold                                     | `otsu_threshold` + `apply_expert_bias_v6`                                                                                                                                                                                                                                                | No manual threshold                                                                | Находит threshold по 1D scores, максимизируя between-class variance. Positive layers выбираются по `pos_auto_score >= pos_otsu_threshold`.                               | Parameter-free selection.                                                                                                     | На малом числе слоёв threshold может быть нестабилен.                                                                                              | `Automatic Otsu layer selection`.                                                                                     |
| Otsu threshold with multiplier for negative layers | Исторический experiment label `negative Otsu x2`; текущий код реализует не multiplier, а double Otsu stage1+stage2.                                                                                                                                      | Исторически `x2`; текущий код: stage1/stage2                | Делает negative layer selection строже.                                                                                                                                                         | Снижает риск чрезмерного negative bias.                                                                 | Название эксперимента и текущий код не полностью совпадают.                                                                         | `Automatic Otsu layer selection + negative Otsu x2`, `new loss + automatic Otsu layer selection, negative Otsu x2`. |
| Automatic layer selection by `layer_score`       | `apply_expert_bias_v6`                                                                                                                                                                                                                                                                     | `eps`, `gap`, selected expert means, Otsu                                      | Считает `pos_auto_score`/`neg_auto_score`, затем выбирает layers через Otsu.                                                                                                   | Учитывает и importance, и router sensitivity.                                                                      | Зависит от качества calibration и answer extraction.                                                                                                                 | Активный default `v6`.                                                                                        |
| Manual positive/negative layer allowlists          | Global `POSITIVE_BIAS_LAYERS`, `NEGATIVE_BIAS_LAYERS`                                                                                                                                                                                                                                    | `None`, list, or `[]`                                                          | Если list задан, слой разрешён только если его индекс в list.`None` включает automatic selection. `[]` запрещает все слои для sign. | Позволяет воспроизводить конкретные debug layer sets.                                        | Нужно править код, CLI параметров нет.                                                                                                                     | Debug comments содержат наборы для math/physics/history/biology/chemistry.                             |
| New loss variants                                  | `compute_expert_importance` old loss commented, new CE active                                                                                                                                                                                                                              | loss function                                                                      | Меняет градиент, а значит expert importance и layer scores.                                                                                                                           | CE over A/B/C/D чище соответствует classification.                                                           | Старые результаты с old loss напрямую не сравнимы с new loss.                                                                                      | Все experiments с prefix `new loss`.                                                                              |

## 11. Expert selection strategies

### Current active expert selection in `v6`

Внутри выбранного слоя `v6` использует adaptive expert selection:

```text
Положительные эксперты внутри выбранного слоя:
importance >= mean(positive_importances_on_layer)

Отрицательные эксперты внутри выбранного слоя:
abs(importance) >= mean(abs(negative_importances_on_layer)) + std(abs(negative_importances_on_layer))
```

Технически:

```python
pos_indices = torch.where(importances > 0)[0]
pos_vals = importances[pos_indices]
pos_layer_mean_threshold = pos_vals.mean().item()
selected_pos = pos_indices[pos_vals >= pos_layer_mean_threshold]

neg_indices = torch.where(importances < 0)[0]
neg_abs_vals = importances[neg_indices].abs()
neg_layer_threshold = neg_abs_vals.mean() + neg_abs_vals.std(unbiased=False)
selected_neg = neg_indices[neg_abs_vals >= neg_layer_threshold]
```

### Historical expert selection variants

| Strategy                       | Status                                                                                                           | Logic                                                                                                                                                                                             | Notes                                                                                               |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| Top-5 experts                  | Historical / original debug support                                                                              | В `apply_expert_bias` печатаются top-5 и bottom-5 normalized importances; экспериментальная таблица содержит `Top-5 experts`.                     | Текущий `v6` не выбирает fixed top-5.                                            |
| Top-3 experts                  | Historical and commented in `v6`                                                                               | В `v6` старый fixed top-3 block сохранён комментарием: выбрать top `NUM_EXP_POS_BIAS_APPLY=3` positive и top `NUM_EXP_NEG_BIAS_APPLY=3` negative by abs. | Многие таблицы старых экспериментов используют `top-3`. |
| Top-1 expert                   | Historical                                                                                                       | Выбирался только самый важный expert.                                                                                                                                   | В текущем коде нет отдельной ветки.                                    |
| Adaptive expert selection      | Active in `v6`                                                                                                 | Positive: above mean positive. Negative: abs above mean+std.                                                                                                                                      | Текущий default.                                                                             |
| Positive experts only / NO neg | Historical/manual                                                                                                | Отключить negative experts, вероятно через `NEGATIVE_BIAS_LAYERS=[]` или старую ветку.                                                                      | Experiment:`top-3+top-3 pos imp layers NO neg`.                                                   |
| With negative experts          | Active if selected negative layers non-empty                                                                     | Negative selected experts получают negative delta bias.                                                                                                                                   | Default `v6` допускает negative experts.                                                 |
| Selection by mean importance   | Active inside `v6` for positive experts                                                                        | `importance >= mean(positive_importances_on_layer)`.                                                                                                                                            | Не fixed top-k.                                                                                   |
| Selection by `mean + std`    | Active inside `v6` for negative experts                                                                        | `abs(importance) >= mean(abs(negative)) + std(abs(negative))`.                                                                                                                                  | Более строгая negative selection.                                                       |
| Selection by quantiles         | Historical/commented layer selection; expert-level quantile logic в текущем коде не активна | Quantile labels относятся в основном к layer scores.                                                                                                                           | Требует проверки старых версий.                                          |

## 12. Bias / routing modification mechanism

### Where the modification happens

Bias применяется после calibration и до evaluation:

```python
model = run_calibration_method(model, expert_importance, mlp_inputs, args)
```

В `run_calibration_method()` выбирается одна из функций `apply_expert_bias*`.

Для `v6`:

```python
sparse_importances = torch.zeros_like(importances)
sparse_importances[selected_indices] = importances[selected_indices]
delta_bias = eps * sparse_importances
layer.mlp.gate.bias += delta_bias
```

### Is this router logits bias or another modification?

Это именно bias к router/gate linear layer. Код не меняет:

- expert weights;
- attention weights;
- embeddings;
- logits head;
- sampling/generation parameters, кроме стандартного `do_sample=False`.

### Runtime or permanent?

Runtime only. Checkpoint не сохраняется. Следующий запуск загрузит исходную модель.

### Positive and negative experts

В `v6` positive и negative experts выбираются отдельно:

- positive experts попадают в `selected_pos_final`, если слой разрешён positive layer selection;
- negative experts попадают в `selected_neg_final`, если слой разрешён negative layer selection;
- затем индексы объединяются;
- знак delta определяется знаком normalized importance.

### What happens to router top-k after bias

После изменения `layer.mlp.gate.bias` каждый следующий forward/generation call использует изменённые router logits. Если bias меняет порядок logits вокруг top-k boundary, expert может попасть или выпасть из top-k. Это и есть механизм изменения поведения MoE.

## 13. Configurable parameters

| Parameter                            | Where defined                                  | Meaning                                                             | Typical values                                                                          | Effect                                                                                                                                                                                                                                                                                                                                                                                                                 |
| ------------------------------------ | ---------------------------------------------- | ------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `SEED`                             | `main.py` global                             | Seed для Python, NumPy, Torch и dataset shuffle.                | `1235`                                                                                | Делает выбор calibration subsets и генерацию deterministic насколько возможно.                                                                                                                                                                                                                                                                                                   |
| `data_folder`                      | `main.py` global                             | Путь к cache MMLU datasets.                                    | `/home/anmilka/MoE/mmlu_data`                                                         | Где читаются/сохраняются datasets.                                                                                                                                                                                                                                                                                                                                                               |
| model path                           | `load_model_and_tokenizer()`                 | Локальный путь `qwen-moe-chat`.                      | `/mnt/storage00/usr/ishukin/models/qwen-moe-chat`                                     | Какая модель загружается.                                                                                                                                                                                                                                                                                                                                                                        |
| `--method`                         | CLI argparse                                   | Bias/evaluation method.                                             | `baseline`, `original`, `v4`, `v5`, `v6`, `v7`, `default`; default `v6` | Выбирает алгоритм router modification.                                                                                                                                                                                                                                                                                                                                                                 |
| `--mode`                           | CLI argparse                                   | Single domain или all domains.                                   | `single`, `all`; default `single`                                                 | `all` запускает subprocess per domain.                                                                                                                                                                                                                                                                                                                                                                      |
| `--domain`                         | CLI argparse                                   | Домен для single mode.                                      | `math`, `physics`, `history`, `biology`, `chemistry`; default `math`        | Выбирает calibration/evaluation datasets.                                                                                                                                                                                                                                                                                                                                                                      |
| `--loss_top_k`                     | CLI argparse and `compute_expert_importance` | Сколько routed experts учитывать при importance. | `top_k`, `top_k_x2`, `all`; default CLI `top_k`, function default `top_k_x2`  | Меняет набор experts, по которым суммируется importance.                                                                                                                                                                                                                                                                                                                                |
| `--eps_value`                      | CLI argparse and `apply_expert_bias_default` | Fixed epsilon for `default` method.                               | `0.005`, `0.01`, `0.05`, `0.1`, `0.2`, `0.3`                                | Сила bias для fixed-eps экспериментов.                                                                                                                                                                                                                                                                                                                                                             |
| `BIAS_START_LAYER`                 | `main.py` global                             | Layer gating.                                                       | `50` current                                                                          | В `v6/original`: `layer_idx >= BIAS_START_LAYER` skip, поэтому с 24-layer model значение 50 обрабатывает все слои. В `v4/v5/v7/default` условие противоположное (`layer_idx < BIAS_START_LAYER: continue`), поэтому при 50 они фактически пропускают все 24 слоя. Это важная асимметрия. |
| `POSITIVE_BIAS_LAYERS`             | `main.py` global                             | Manual allowlist positive layers.                                   | `None`, list, `[]`                                                                  | `None` включает auto selection; list ограничивает; `[]` отключает positive bias.                                                                                                                                                                                                                                                                                                      |
| `NEGATIVE_BIAS_LAYERS`             | `main.py` global                             | Manual allowlist negative layers.                                   | `None`, list, `[]`                                                                  | Аналогично для negative bias.                                                                                                                                                                                                                                                                                                                                                                             |
| `NUM_EXP_POS_BIAS_APPLY`           | `main.py` global/commented v6 block          | Старый fixed top-k count positive experts.                    | `3`                                                                                   | В текущем active `v6` не используется, но отражает старую top-3 логику.                                                                                                                                                                                                                                                                                                  |
| `NUM_EXP_NEG_BIAS_APPLY`           | `main.py` global/commented v6 block          | Старый fixed top-k count negative experts.                    | `3`                                                                                   | В текущем active `v6` не используется.                                                                                                                                                                                                                                                                                                                                                         |
| `POSITIVE_TOP3_MEAN_THRESHOLD`     | `main.py` global                             | Historical threshold constant.                                      | `0.6`                                                                                 | В текущей active ветке не используется.                                                                                                                                                                                                                                                                                                                                                     |
| `NEGATIVE_TOP3_ABS_MEAN_THRESHOLD` | `main.py` global                             | Historical negative threshold constant.                             | `0.6`                                                                                 | В текущей active ветке не используется.                                                                                                                                                                                                                                                                                                                                                     |
| `generation_max_new_tokens`        | `compute_expert_importance` arg              | Max generation tokens during calibration.                           | `20`                                                                                  | Ограничивает поиск answer-bearing generation step.                                                                                                                                                                                                                                                                                                                                                    |
| evaluation `max_new_tokens`        | `generate_responses` arg                     | Max generation tokens during accuracy evaluation.                   | `512`                                                                                 | Длина ответа при evaluation.                                                                                                                                                                                                                                                                                                                                                                             |
| `batch_size`                       | `test_accuracy` arg                          | Batch size для evaluation.                                       | `16`                                                                                  | Влияет на память и скорость evaluation.                                                                                                                                                                                                                                                                                                                                                         |
| `top_k`                            | `layer.mlp.top_k` or model config            | Router experts per token.                                           | model-dependent, often `4`                                                            | Используется в routing and gap computation.                                                                                                                                                                                                                                                                                                                                                               |
| `eps`                              | computed inside bias methods                   | Bias scale.                                                         | method-dependent                                                                        | Масштабирует normalized importances into gate bias deltas.                                                                                                                                                                                                                                                                                                                                                 |
| `gap` / `gk`                     | `apply_expert_bias_v6`                       | Mean router logit margin top-k vs top-k+1.                          | computed                                                                                | Используется в layer score and eps.                                                                                                                                                                                                                                                                                                                                                                       |
| `pos_q`, `neg_q`                 | historical/commented quantile experiments      | Quantile thresholds for layer score.                                | `0.50`, `0.60`, `0.70`, `0.80`, `0.90`                                        | В текущем code inactive; использовались в старых экспериментах.                                                                                                                                                                                                                                                                                                              |
| Otsu threshold                       | `otsu_threshold`                             | Auto threshold for layer scores.                                    | computed                                                                                | Active `v6` positive layer selection.                                                                                                                                                                                                                                                                                                                                                                                |
| Negative double Otsu                 | `apply_expert_bias_v6`                       | Two-stage Otsu for negative layer scores.                           | computed                                                                                | Active `v6` negative layer selection.                                                                                                                                                                                                                                                                                                                                                                                |
| output path                          | отсутствует                         | Сохранение результатов.                        | N/A                                                                                     | Текущий код печатает результаты, но не сохраняет их.                                                                                                                                                                                                                                                                                                                        |

## 14. How to run

Команды восстановлены по argparse и текущему `main.py`.

### Baseline evaluation

```bash
python main.py --method baseline
```

Важное caveat: baseline block в текущем коде вручную закомментирован/раскомментирован. Сейчас активны только:

```text
high_school_chemistry
college_chemistry
```

Кроме того, после baseline block код не завершает процесс и дальше может перейти к calibration. Для чистого baseline это поведение нужно проверить или исправить вручную.

### Calibration + modified routing for one domain

```bash
python main.py --method v6 --mode single --domain math --loss_top_k top_k
```

Другие домены:

```bash
python main.py --method v6 --mode single --domain physics --loss_top_k top_k
python main.py --method v6 --mode single --domain history --loss_top_k top_k
python main.py --method v6 --mode single --domain biology --loss_top_k top_k
python main.py --method v6 --mode single --domain chemistry --loss_top_k top_k
```

### Expert importance collection

Отдельной CLI-команды "только собрать importance" нет. Importance собирается внутри любого non-baseline run:

```bash
python main.py --method v6 --mode single --domain math --loss_top_k top_k
```

Если нужно только collection без evaluation/modification, текущий код надо доработать.

### Modified routing / biased experts evaluation

Default/current method:

```bash
python main.py --method v6 --mode single --domain math --loss_top_k top_k
```

С учётом расширенного набора experts для importance:

```bash
python main.py --method v6 --mode single --domain physics --loss_top_k top_k_x2
python main.py --method v6 --mode single --domain physics --loss_top_k all
```

### Run all domains

```bash
python main.py --method v6 --mode all --loss_top_k top_k
```

`--mode all` запускает отдельный Python subprocess на каждый домен. Это сделано, чтобы не держать CUDA memory между доменами.

### Fixed epsilon sweep

```bash
python main.py --method default --loss_top_k top_k
```

Текущий код в этом режиме запускает sweep:

```text
0.005, 0.01, 0.05, 0.1, 0.2, 0.3
```

Эта ветка math-centric: importance считается на `test_samples_math`, evaluation на `college_mathematics`, `abstract_algebra`, `all_part`.

Для `default` без sweep:

```bash
python main.py --method default --loss_top_k top_k_x2 --eps_value 0.05
```

## 15. How to reproduce experiments

Общий порядок воспроизведения:

1. Убедиться, что локальная модель доступна по пути `/mnt/storage00/usr/ishukin/models/qwen-moe-chat`.
2. Убедиться, что доступны зависимости: `torch`, `transformers`, `datasets`, `numpy`, `tqdm`.
3. Убедиться, что есть доступ к cache `/home/anmilka/MoE/mmlu_data` или к Hugging Face dataset `cais/mmlu`.
4. Зафиксировать `SEED = 1235`.
5. Выбрать домен через `--domain`.
6. Выбрать `--loss_top_k`.
7. Для текущего основного эксперимента запустить `--method v6`.
8. Скопировать stdout debug output: там печатаются выбранные layers, thresholds, experts и final accuracy.

Для старых экспериментов из таблицы часть логики не представлена как CLI switches. Для их точного воспроизведения нужно менять constants или восстановить старые ветки:

- fixed top-5/top-3/top-1 experts;
- fixed first 4 layers;
- top-N layers by importance;
- thresholds `pos_imp>0.6`, `neg_imp>0.7`;
- quantiles `pos_q`, `neg_q`;
- old loss.

Текущий `main.py` лучше всего воспроизводит эксперименты семейства:

- `new loss + automatic Otsu layer selection, negative Otsu x2`;
- adaptive expert selection by layer mean/mean+std;
- `layer-score = (eps * mean(selected_importance)) / gap`.

## 16. How to add a new experiment

Минимальный путь:

1. Добавить новый method name в `parser.add_argument('--method', choices=[...])`.
2. Реализовать новую функцию `apply_expert_bias_<name>(model, expert_importance, all_router_logits)`.
3. Добавить ветку в `run_calibration_method()`.
4. Если нужен новый domain, добавить entry в `get_domain_configs()`.
5. Если нужно менять layer/expert selection без правки кода, лучше добавить CLI args вместо новых globals.
6. Для воспроизводимости печатать:
   - calibration dataset;
   - loss type;
   - selected positive layers;
   - selected negative layers;
   - selected experts per layer;
   - thresholds;
   - final accuracy.

Рекомендуемые улучшения перед продолжением:

- добавить `--model_path`, `--data_folder`, `--output_path`;
- добавить `--calibration_samples`;
- добавить CLI для `POSITIVE_BIAS_LAYERS` и `NEGATIVE_BIAS_LAYERS`;
- добавить режим `--collect_importance_only`;
- сохранять JSON/CSV с results и debug layer stats;
- явно отделить baseline path от calibration path.

## 17. Known limitations and caveats

### Unclear or not fully recoverable from code

- В текущем репозитории нет старых версий кода, поэтому эксперименты `Top-5 experts`, `Top-3 experts`, `Top-1 expert`, `First 4 layers only`, `top-3+3 biggest exp imp layers` восстановлены по таблице результатов и комментариям, но не могут быть полностью подтверждены текущей active реализацией.
- `pos_q`/`neg_q` quantile layer selection есть в experiment log и частично в закомментированном block `apply_expert_bias_v6`, но переменные quantile threshold отсутствуют в текущем коде.
- Experiment label `negative Otsu x2` исторически может означать multiplier для negative threshold, но текущий код реализует double Otsu stage1+stage2, а не умножение threshold на 2.
- Протокол таблицы говорит, что calibration dataset всегда 100 samples. В текущем коде для `math` используется `test_samples_math` без `.select(range(100))`, а для `college_physics` объект `test_samples_college_physics` уже урезан до 100 и используется также в evaluation. Это требует проверки относительно фактических запусков, давших таблицу.
- Baseline branch в `main.py` не завершает процесс после baseline evaluation. Поэтому `python main.py --method baseline` может выполнить последующую calibration logic. Это выглядит как поведенческий баг или остаток от ручного режима.
- `BIAS_START_LAYER` используется асимметрично: `v6`/`original` обрабатывают слои ниже threshold, а `v4/v5/v7/default` - слои выше или равные threshold. При текущем `BIAS_START_LAYER=50` и 24-layer model `v4/v5/v7/default` могут фактически ничего не изменить.
- Результаты не сохраняются автоматически; experiment table в этом документе взята из запроса пользователя, а не из файлов `outputs/` или `results/`.

### Legacy / experimental scripts

В текущем checkout нет отдельных legacy scripts. В `CODEX.md` упоминается `check_router_weights.py`, но файла нет в проекте. Legacy/experimental logic находится внутри `main.py` как:

- `apply_expert_bias`;
- `apply_expert_bias_v4`;
- `apply_expert_bias_v5`;
- `apply_expert_bias_v7`;
- `apply_expert_bias_default`;
- закомментированный old loss;
- закомментированный fixed top-3 expert selection;
- закомментированный quantile layer-selection block;
- закомментированные baseline datasets.

## 18. Experiment log and results

Ниже структурирована таблица экспериментов из задания.

Важно:

- `*` рядом с названием датасета означает, что этот датасет использовался для калибровки.
- Размер calibration dataset всегда 100 samples.
- Для тестирования качества всегда использовался оригинальный полный размер датасета.
- Например, если `high_school_world_history = 268 samples`, для калибровки брались random 100 samples, а evaluation шёл на всех 268 samples.
- Пустые ячейки означают, что результат не был указан в предоставленной таблице.

### Experiment descriptions

| Experiment                                                                                             | What changed / hypothesis                                                                            | Layers                                                                                 | Experts                                                     | Loss / thresholds                                          | Result interpretation                                                                                                                                  |
| ------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | ----------------------------------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Baseline                                                                                               | Без router modification.                                                                          | None                                                                                   | None                                                        | Standard generation accuracy.                              | Reference quality.                                                                                                                                     |
| Top-5 experts                                                                                          | Проверка широкого выбора наиболее важных экспертов.     | Вероятно все или выбранные старой логикой.         | Top-5 by importance.                                        | Old experimental logic.                                    | На math calibration улучшает `college_mathematics`, но не даёт стабильного выигрыша на всех math subsets. |
| Top-3 experts                                                                                          | Более sparse expert bias.                                                                       | Вероятно все или выбранные старой логикой.         | Top-3 by importance.                                        | Old experimental logic.                                    | Math results близки к top-5;`elementary_mathematics` выше baseline, `high_school_mathematics` ниже.                                 |
| Top-1 expert                                                                                           | Максимально sparse intervention.                                                          | Вероятно все или выбранные старой логикой.         | Top-1 by importance.                                        | Old experimental logic.                                    | Не показывает явного преимущества над top-3.                                                                          |
| First 4 layers only                                                                                    | Проверка гипотезы о важности ранних слоёв.                       | First 4 layers.                                                                        | Не указано, вероятно selected top experts. | Old experimental logic.                                    | На `college_mathematics` выше baseline; на других mixed.                                                                               |
| top-3+4layers                                                                                          | Комбинация fixed first layers и top-3 experts.                                            | 4 layers.                                                                              | Top-3.                                                      | Old experimental logic.                                    | Похожий результат на first 4 layers;`high_school_mathematics` проседает.                                                  |
| top-3+3 biggest exp imp layers                                                                         | Выбор 3 слоёв с наибольшей expert importance.                                   | Top-3 layers by expert importance.                                                     | Top-3.                                                      | Old layer scoring.                                         | Один из лучших math/history вариантов, но physics почти baseline.                                                          |
| top-3+5 biggest exp imp layers                                                                         | Расширение до 5 важных слоёв.                                                 | Top-5 layers by importance.                                                            | Top-3.                                                      | Old layer scoring.                                         | Не улучшает устойчиво относительно top-3 layers.                                                                        |
| top-3+mean imp > 0.6                                                                                   | Threshold по mean importance.                                                                      | Layers above `0.6`.                                                                  | Top-3.                                                      | Threshold `0.6`.                                         | Даёт небольшие локальные улучшения, но без универсального эффекта.                            |
| top-3+pos_imp>0.6,neg_imp>0.7                                                                          | Отдельные thresholds для positive/negative.                                              | Positive and negative thresholded.                                                     | Top-3 positive/negative.                                    | `pos>0.6`, `neg>0.7`.                                  | Учитывает sign asymmetry; результаты близки к mean threshold.                                                                |
| top-3+top-3 pos imp layers NO neg                                                                      | Проверка positive-only bias без negative suppression.                                     | Top positive layers.                                                                   | Positive top-3, no negative.                                | Negative disabled.                                         | Часто близко к baseline; показывает, что negative experts не всегда нужны.                                       |
| `(eps * mean(top3)) / gap, pos_q=0.70 neg_q=0.80`                                                    | Layer score учитывает importance, eps и gap; quantile thresholds.                          | Quantile-selected.                                                                     | Top-3.                                                      | `pos_q=0.70`, `neg_q=0.80`.                            | Math mixed, не лучше baseline по `college_mathematics`.                                                                                     |
| `(eps * mean(top3)) / gap, pos_q=0.60 neg_q=0.90`                                                    | Более широкий positive selection и строгий negative selection.                   | Quantile-selected.                                                                     | Top-3.                                                      | `pos_q=0.60`, `neg_q=0.90`.                            | Math ближе к baseline; physics `high_school_physics` ухудшается.                                                                     |
| `(eps * mean(top3)) / gap, pos_q=0.50 neg_q=0.90`                                                    | Ещё шире positive layers.                                                                     | Positive layers `[0,1,2,4,6,7,17,19,20,21,22,23]`, negative `[4,5,18]` from debug. | Top-3.                                                      | `pos_q=0.50`, `neg_q=0.90`.                            | Math `all_part` сохраняет baseline 0.60, но `college_mathematics` ниже baseline.                                                    |
| `(eps * mean(top3)) / gap, pos_threshold=mean(pos_scores), neg_threshold=mean+std`                   | Absolute/statistical layer thresholds по scores.                                                   | Threshold-selected.                                                                    | Top-3.                                                      | `pos_threshold=0.22644436`, `neg_threshold=0.242036`.  | Результаты совпадают с предыдущим указанным math set.                                                           |
| `(eps * mean(top3)) / gap, importance >= mean(positive_importances_on_layer), pos_q=0.60 neg_q=0.90` | Переход от fixed top-3 experts к adaptive positive expert selection внутри слоя. | Quantile-selected.                                                                     | Positive above layer mean; negative by quantile/old logic.  | `pos_q=0.60`, `neg_q=0.90`.                            | Math `college_mathematics` выше baseline, physics `high_school_physics` ниже.                                                              |
| Automatic Otsu layer selection                                                                         | Parameter-free layer thresholding by Otsu on layer score.                                            | Otsu-selected.                                                                         | Adaptive: positive above mean, negative abs above mean+std. | `layer-score = (eps * mean(selected_importance)) / gap`. | Math close to baseline; physics slightly mixed.                                                                                                        |
| Automatic Otsu layer selection + negative Otsu x2                                                      | Более строгий negative selection.                                                        | Otsu positive, stricter negative.                                                      | Adaptive.                                                   | Historical label; current code uses double Otsu.           | Небольшие изменения без устойчивого доминирования.                                                        |
| new loss + layer select by mean imp > 0.6 + top-3 experts                                              | Проверка CE loss вместо old mixed loss при threshold layer selection.               | Positive `[13,22]`, negative `[2,15,17,19]` from debug.                            | Top-3.                                                      | New CE, mean imp threshold.                                | Physics `high_school_physics` 0.39 выше baseline 0.38.                                                                                           |
| new loss + automatic Otsu layer selection, negative Otsu x2                                            | Текущая основная логика: CE + Otsu/double Otsu + adaptive experts.              | Positive `[0,1,6,7,20,21,22,23]`, negative `[2,4,5,19]` from debug.                | Adaptive positive and negative.                             | New CE, Otsu/double Otsu.                                  | Улучшает `conceptual_physics` до 0.52, math mostly near baseline.                                                                          |
| new loss +`(eps * mean(top3)) / gap`, `pos_q=0.60 neg_q=0.90`                                      | CE + quantile score selection.                                                                       | Positive `[0,1,4,6,7,9,20,21,22,23]`, negative `[2,17,19]` from debug.             | Top-3/adaptive mixed by experiment label.                   | New CE, quantiles.                                         | Physics `high_school_physics` ухудшается до 0.35.                                                                                        |

### Mathematics results

| Experiment                                                               | college_mathematics* | abstract_algebra | elementary_mathematics | high_school_mathematics | all_part |
| ------------------------------------------------------------------------ | -------------------: | ---------------: | ---------------------: | ----------------------: | -------: |
| Baseline                                                                 |                 0.38 |             0.39 |                   0.40 |                    0.37 |     0.60 |
| Top-5 experts                                                            |                 0.41 |             0.38 |                   0.40 |                    0.36 |     0.59 |
| Top-3 experts                                                            |                 0.41 |             0.38 |                   0.42 |                    0.34 |     0.58 |
| Top-1 expert                                                             |                 0.39 |             0.38 |                   0.42 |                    0.34 |     0.58 |
| First 4 layers only                                                      |                 0.42 |             0.38 |                   0.42 |                    0.35 |     0.59 |
| top-3+4layers                                                            |                 0.42 |             0.39 |                   0.42 |                    0.33 |     0.59 |
| top-3+3 biggest exp imp layers                                           |                 0.42 |             0.41 |                   0.42 |                    0.38 |     0.60 |
| top-3+5 biggest exp imp layers                                           |                 0.41 |             0.40 |                   0.41 |                    0.36 |          |
| top-3+mean imp > 0.6                                                     |                 0.42 |             0.41 |                   0.41 |                    0.38 |     0.60 |
| top-3+pos_imp>0.6,neg_imp>0.7                                            |                 0.42 |             0.41 |                   0.41 |                    0.38 |     0.60 |
| top-3+top-3 pos imp layers NO neg                                        |                 0.41 |             0.40 |                   0.41 |                    0.38 |     0.60 |
| `(eps * mean(top3)) / gap, pos_q=0.70 neg_q=0.80`                      |                 0.38 |             0.40 |                   0.42 |                    0.37 |     0.59 |
| `(eps * mean(top3)) / gap, pos_q=0.60 neg_q=0.90`                      |                 0.38 |             0.41 |                   0.41 |                    0.38 |     0.60 |
| `(eps * mean(top3)) / gap, pos_q=0.50 neg_q=0.90`                      |                 0.39 |             0.41 |                   0.42 |                    0.35 |     0.60 |
| `(eps * mean(top3)) / gap, pos_threshold=mean, neg_threshold=mean+std` |                 0.39 |             0.41 |                   0.42 |                    0.35 |     0.60 |
| `(eps * mean(top3)) / gap, adaptive experts, pos_q=0.60 neg_q=0.90`    |                 0.41 |             0.39 |                   0.42 |                    0.36 |     0.60 |
| Automatic Otsu layer selection                                           |                 0.40 |             0.39 |                   0.42 |                    0.37 |     0.59 |
| Automatic Otsu layer selection + negative Otsu x2                        |                 0.40 |             0.41 |                   0.41 |                    0.37 |     0.60 |
| new loss + automatic Otsu layer selection, negative Otsu x2              |                 0.40 |             0.39 |                   0.42 |                    0.36 |     0.60 |

### Physics results

| Experiment                                                            | college_physics* | conceptual_physics | high_school_physics | all_part |
| --------------------------------------------------------------------- | ---------------: | -----------------: | ------------------: | -------: |
| Baseline                                                              |             0.35 |               0.51 |                0.38 |     0.60 |
| top-3+3 biggest exp imp layers                                        |             0.36 |               0.51 |                0.38 |          |
| top-3+5 biggest exp imp layers                                        |             0.36 |               0.50 |                0.36 |          |
| top-3+mean imp > 0.6                                                  |             0.36 |               0.51 |                0.38 |          |
| top-3+pos_imp>0.6,neg_imp>0.7                                         |             0.36 |               0.51 |                0.37 |     0.59 |
| top-3+top-3 pos imp layers NO neg                                     |             0.35 |               0.51 |                0.38 |     0.60 |
| `(eps * mean(top3)) / gap, pos_q=0.60 neg_q=0.90`                   |             0.35 |               0.51 |                0.34 |          |
| `(eps * mean(top3)) / gap, adaptive experts, pos_q=0.60 neg_q=0.90` |             0.36 |               0.51 |                0.35 |          |
| Automatic Otsu layer selection                                        |             0.36 |               0.50 |                0.36 |          |
| Automatic Otsu layer selection + negative Otsu x2                     |             0.35 |               0.51 |                0.37 |          |
| new loss + layer select by mean imp > 0.6 + top-3 experts             |             0.35 |               0.51 |                0.39 |          |
| new loss + automatic Otsu layer selection, negative Otsu x2           |             0.35 |               0.52 |                0.38 |          |
| new loss +`(eps * mean(top3)) / gap, pos_q=0.60 neg_q=0.90`         |             0.35 |               0.51 |                0.35 |          |

### History results

| Experiment                        | high_school_world_history* | high_school_european_history | high_school_us_history | prehistory | all_part |
| --------------------------------- | -------------------------: | ---------------------------: | ---------------------: | ---------: | -------: |
| Baseline                          |                       0.72 |                         0.69 |                   0.69 |       0.65 |     0.60 |
| top-3+3 biggest exp imp layers    |                       0.72 |                         0.69 |                   0.70 |       0.65 |          |
| top-3+5 biggest exp imp layers    |                       0.72 |                         0.68 |                   0.70 |       0.65 |          |
| top-3+mean imp > 0.6              |                       0.73 |                         0.68 |                   0.71 |       0.65 |     0.59 |
| top-3+pos_imp>0.6,neg_imp>0.7     |                       0.72 |                         0.69 |                   0.71 |       0.65 |     0.59 |
| top-3+top-3 pos imp layers NO neg |                       0.71 |                         0.69 |                   0.70 |       0.65 |     0.59 |

### Biology results

| Experiment                                        | college_biology* | high_school_biology | all_part |
| ------------------------------------------------- | ---------------: | ------------------: | -------: |
| Baseline                                          |             0.58 |                0.69 |     0.60 |
| top-3+5 biggest exp imp layers                    |             0.60 |                0.68 |     0.60 |
| top-3+mean imp > 0.6                              |             0.61 |                0.69 |     0.59 |
| top-3+pos_imp>0.6,neg_imp>0.7                     |             0.60 |                0.68 |     0.60 |
| top-3+top-3 pos imp layers NO neg                 |             0.59 |                0.68 |     0.60 |
| Automatic Otsu layer selection + negative Otsu x2 |             0.60 |                0.69 |          |

### Chemistry results

| Experiment                                        | college_chemistry* | high_school_chemistry | all_part |
| ------------------------------------------------- | -----------------: | --------------------: | -------: |
| Baseline                                          |               0.45 |                  0.45 |     0.60 |
| top-3+5 biggest exp imp layers                    |               0.49 |                  0.44 |     0.58 |
| top-3+mean imp > 0.6                              |               0.49 |                  0.43 |     0.58 |
| top-3+pos_imp>0.6,neg_imp>0.7                     |               0.49 |                  0.45 |     0.59 |
| top-3+top-3 pos imp layers NO neg                 |               0.50 |                  0.45 |     0.60 |
| Automatic Otsu layer selection + negative Otsu x2 |               0.48 |                  0.44 |          |
