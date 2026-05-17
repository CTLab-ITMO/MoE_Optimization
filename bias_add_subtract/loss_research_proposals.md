# Loss research proposals

## 1. Context from current project

Проект `WES` исследует runtime-модификацию MoE routing в `qwen-moe-chat`: на calibration dataset считается signed expert importance, затем selected experts/layers получают sparse bias в `layer.mlp.gate.bias`, после чего модель оценивается на MMLU-подобных датасетах.

Фактическая реализация находится в `main.py`.

Ключевые участки:

| Area                                    | File / function                                                                     | Role                                                                                                                                                |
| --------------------------------------- | ----------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| Текущий loss                     | `main.py`, `compute_expert_importance`                                          | Генерация ответа, поиск answer step, сбор `A/B/C/D` logits, CE loss, backward.                                            |
| Извлечение logits `A/B/C/D` | `main.py`, `get_letter_token_ids`, `build_abcd_logits`                        | Находит single-token ids для букв и собирает 4 answer logits из full vocab logits.                                         |
| Expert importance                       | `main.py`, `compute_expert_importance`                                          | Использует градиент loss по MLP output и вклад эксперта `w_i * h_i`: `importance = -dot(g_token, w_i * h_i)`. |
| Positive/negative importance            | `main.py`, `apply_expert_bias_v6`                                               | Нормализует importances, разделяет `>0` и `<0`, выбирает experts по mean / mean+std.                             |
| Layer selection                         | `main.py`, `apply_expert_bias_v6`, `otsu_threshold`                           | Считает `pos_auto_score`, `neg_auto_score`, выбирает layers через Otsu / double Otsu или manual allowlists.              |
| Bias application                        | `main.py`, `apply_expert_bias_v6` и legacy `apply_expert_bias*`              | Создаёт/обновляет `layer.mlp.gate.bias`, добавляет `delta_bias = eps * sparse_importances`.                            |
| Evaluation                              | `main.py`, `generate_responses`, `test_accuracy`, `evaluate_named_datasets` | Генерация ответов и exact-match accuracy по `A/B/C/D`.                                                                         |

В момент подсчёта loss в `compute_expert_importance` доступны:

- `example["answer"]` и `correct_idx`;
- `generated_ids`, `generated_text`, `extracted_gen_idx`, `extracted_answer`;
- `conditioning_ids = prompt + generated_prefix_before_answer`;
- `pred_pos`, позиция logits, предсказывающая answer token;
- `outputs.logits[0, pred_pos, :]` как full vocab logits;
- `abcd_logits` и `probs_abcd`;
- MLP input per layer через forward hooks: `mlp_inputs[layer_idx]`;
- gradient per layer через backward hooks: `hidden_states_with_grad[layer_idx]`;
- `router_logits = layer.mlp.gate(hidden_flat)`;
- `routing_weights = softmax(router_logits)`;
- `selected` expert ids and `weights` for `top_k`, `top_k_x2` или `all`;
- `layer.mlp.experts[expert_idx]`, чтобы пересчитать expert output `h_i` на нужной позиции.

Текущие ограничения:

- loss считается только на одной token position;
- skipped samples возникают, если `A/B/C/D` не извлечена за первые 20 generated tokens;
- результаты calibration не сохраняются как structured artifact;
- baseline logits до/после bias в одной функции сейчас не сравниваются;
- router/expert losses сейчас не используются как differentiable objective, а только как post-hoc scoring можно добавить поверх доступных tensors;
- causal/delta losses потребуют дополнительных forward passes или временных expert/router interventions.

## 2. Current loss logic

Текущий алгоритм:

```text
1. Модель генерирует до 20 токенов.
2. Находится первый шаг, где удалось извлечь букву ответа A/B/C/D.
3. На этой позиции берутся logits словаря.
4. Из logits извлекаются 4 значения: logits для A, B, C, D.
5. Над этими четырьмя logits считается Cross Entropy относительно gold-answer.
6. Полученный loss используется для оценки влияния экспертов / слоёв.
```

В коде это находится в `main.py`, `compute_expert_importance`:

```python
logits_pos = outputs.logits[0, pred_pos, :]
abcd_logits = build_abcd_logits(logits_pos, letter_token_ids)
probs_abcd = F.softmax(abcd_logits.float(), dim=-1)
target = torch.tensor([correct_idx], device=abcd_logits.device)
loss = F.cross_entropy(abcd_logits.float().unsqueeze(0), target)
loss.backward()
```

Старый mixed loss оставлен комментарием:

```python
loss_correct = -torch.log(p_correct)
loss_wrong = 1.0 - p_correct
beta = (1.0 - p_correct).detach()
loss = loss_correct + beta * loss_wrong
```

Активный вариант сейчас - plain CE over four answer logits.

## 3. Limitations of current loss

Плюсы текущего CE:

- полностью совместим с текущей MMLU постановкой `A/B/C/D`;
- дешёвый: нужен один forward после generation;
- даёт корректный gradient signal для `loss.backward()`;
- проще интерпретировать, чем старый mixed-scale loss;
- напрямую оптимизирует ranking gold answer среди четырёх вариантов.

Слабые места:

- CE видит только четыре answer logits и игнорирует весь остальной словарь. Если модель хочет ответить словами `"The answer is B"` или другим форматом, full-vocab competition не отражается.
- Loss считается на одной позиции `pred_pos`. Если важная reasoning-информация распределена по нескольким generated tokens до буквы, signal слишком узкий.
- Если answer letter появляется поздно, conditioning включает часть generated rationale. Тогда importance отражает не только исходный вопрос, но и уже сгенерированный prefix.
- Если extraction сработал на случайную букву в неправильном формате, loss будет привязан к шумной позиции.
- Samples без extracted answer за 20 tokens пропускаются, что может смещать calibration к примерам, где модель и так отвечает коротко.
- CE saturates: при очень уверенно правильных или уверенно неправильных ответах градиенты могут быть менее полезными для различения экспертов.
- CE не учитывает router stability. Expert может иметь высокий gradient importance, но router gap настолько большой, что bias практически не изменит top-k.
- Текущий gradient-based importance является first-order proxy, не causal measurement. Он не проверяет, что отключение/усиление эксперта реально меняет answer margin.

## 4. Available signals in the current code

Сигналы, которые уже есть без архитектурных изменений:

| Signal                                      | Где доступен                                        | Использование                                             |
| ------------------------------------------- | -------------------------------------------------------------- | ---------------------------------------------------------------------- |
| `abcd_logits`                             | `compute_expert_importance` после `build_abcd_logits` | CE, margin, entropy, answer-only KL.                                   |
| `probs_abcd`                              | `compute_expert_importance`                                  | Confidence weighting, ambiguity weighting.                             |
| `logits_pos` full vocab                   | `compute_expert_importance`                                  | Full/top-N vocab KL, format penalties, teacher distribution.           |
| `correct_idx` / gold label                | `compute_expert_importance`                                  | Supervised losses.                                                     |
| `extracted_answer`, `extracted_gen_idx` | `compute_expert_importance`                                  | Sample filtering/reweighting by answer position and parser confidence. |
| `mlp_inputs[layer_idx]`                   | forward hook                                                   | Recompute expert outputs for importance.                               |
| `hidden_states_with_grad[layer_idx]`      | backward hook                                                  | First-order expert contribution scoring.                               |
| `router_logits`                           | `compute_expert_importance`, per layer                       | Router KL, entropy, gap, top-k stability.                              |
| `routing_weights`                         | `compute_expert_importance`                                  | Expert probability score, entropy, selected expert reward/penalty.     |
| `selected` expert ids                     | `compute_expert_importance`                                  | Top-k overlap, contrastive expert scores.                              |
| `expert_importance`                       | returned to bias methods                                       | Layer/expert aggregation.                                              |
| `all_router_logits`                       | returned to bias methods                                       | Layer gap, eps, Otsu layer selection.                                  |

Сигналы, которые потребуют дополнительного кода:

- baseline vs modified logits on the same examples;
- per-expert ablation/downweight causal effects;
- per-layer intervention effects;
- bootstrap resampling statistics;
- JSON/CSV cache для offline comparison selected layers/experts.

## 5. Loss proposal 1: Answer CE baseline refinement

### Idea

Оставить CE как базовый supervised signal, но сделать его менее шумным через temperature, label smoothing и sample weights. Это минимальное изменение текущей логики.

### Formula

Temperature-scaled CE:

```math
L_{CE-T} = - \log \frac{\exp(z_y / T)}{\sum_{c \in \{A,B,C,D\}} \exp(z_c / T)}
```

Label smoothing:

```math
q_c =
\begin{cases}
1 - \epsilon_{ls}, & c = y \\
\epsilon_{ls}/3, & c \ne y
\end{cases}
```

```math
L_{smooth} = - \sum_c q_c \log p_c
```

Confidence / extraction weighting:

```math
L = w_{sample} \cdot L_{smooth}
```

где, например:

```math
w_{sample} = \min(1, \lambda_1 + \lambda_2 \cdot (1 - p_y))
```

или penalty за позднюю extraction:

```math
w_{sample} = \exp(-\rho \cdot extracted\_gen\_idx)
```

### What signal it captures

Тот же answer-quality signal, что текущий CE, но с меньшей чувствительностью к экстремальным logits и шумным examples.

### Why it may work better

На 100 calibration samples единичные шумные примеры могут сильно влиять на accumulated expert importance. Temperature и label smoothing смягчают gradient spikes; sample weighting позволяет меньше доверять поздним или ambiguous extractions.

### How to implement in current project

В `main.py`, `compute_expert_importance`, заменить строку:

```python
loss = F.cross_entropy(abcd_logits.float().unsqueeze(0), target)
```

на helper вроде:

```python
loss = compute_answer_ce_refined(
    abcd_logits,
    correct_idx,
    temperature=1.5,
    label_smoothing=0.02,
    extracted_gen_idx=extracted_gen_idx,
)
```

Лучше добавить CLI `--loss_type ce_refined`, `--temperature`, `--label_smoothing`, `--position_decay`.

### Required tensors / data

- `abcd_logits`;
- `correct_idx`;
- `extracted_gen_idx`;
- optionally `probs_abcd`.

### Expected effect

Более стабильные expert importance distributions и меньше резких layer-score выбросов на малом calibration set.

### Risks / caveats

- Может ослабить gradient для действительно информативных difficult samples.
- Если label smoothing слишком большой, positive/negative expert separation станет слабее.
- Не решает проблему single-token objective.

### Experiment plan

1. На math/physics calibration сравнить layer/expert overlap с текущим CE.
2. Проверить distribution of importance: max/min, number of selected positive/negative experts.
3. Запустить small evaluation: calibration domain + 1 соседний dataset.

### Priority

High.

## 6. Loss proposal 2: Answer margin loss

### Idea

Вместо вероятностного CE оптимизировать margin между gold logit и самым сильным wrong answer logit.

### Formula

Gold margin:

```math
m = z_y - \max_{c \ne y} z_c
```

Hinge margin loss:

```math
L_{margin} = \max(0, \tau - m)
```

Smooth variant:

```math
L_{soft-margin} = \log(1 + \exp(\tau - m))
```

### What signal it captures

Насколько уверенно gold answer отделён от ближайшего конкурента среди `A/B/C/D`.

### Why it may work better

Evaluation использует exact answer choice. Для выбора экспертов важнее не абсолютная probability, а сможет ли intervention изменить rank между gold и ближайшим wrong answer. Margin loss напрямую оптимизирует это.

Margin может быть стабильнее CE: когда wrong classes слабы, он перестаёт давать сильный signal; когда есть близкий конкурент, gradient концентрируется на relevant boundary.

### How to implement in current project

В `compute_expert_importance` после `abcd_logits`:

```python
gold = abcd_logits[correct_idx]
wrong = torch.cat([abcd_logits[:correct_idx], abcd_logits[correct_idx + 1:]])
max_wrong = wrong.max()
margin = gold - max_wrong
loss = F.softplus(margin_target - margin)
```

Добавить `--loss_type answer_margin` и `--answer_margin_tau`.

### Required tensors / data

- `abcd_logits`;
- `correct_idx`;
- optionally baseline margin for logging.

### Expected effect

Expert importance станет больше отражать experts, которые двигают модель через decision boundary, а не просто меняют calibrated probability.

### Risks / caveats

- Hinge loss с большим flat region может давать нулевой gradient на простых examples.
- Softplus лучше для стабильности, но добавляет hyperparameter `tau`.
- Не учитывает full-vocab competition и format issues.

### Experiment plan

1. Начать с `tau=0.0` и `tau=1.0`.
2. Сравнить selected layers с CE и проверить, уменьшается ли количество noisy negative layers.
3. Запустить на math и physics, потому что там baseline дал наиболее mixed results.

### Priority

High.

## 7. Loss proposal 3: Delta CE causal effect

### Idea

Перейти от first-order gradient proxy к более causal scoring: измерять изменение CE при временном downweight/disable конкретного эксперта или группы экспертов.

### Formula

Для expert `e`:

```math
\Delta L_e = L_{CE}(x; intervention(e)) - L_{CE}(x; baseline)
```

Интерпретация:

```text
Delta L_e > 0: intervention ухудшил CE, expert positive/helpful.
Delta L_e < 0: intervention улучшил CE, expert negative/harmful.
Delta L_e ~= 0: neutral.
```

### What signal it captures

Approximate causal contribution expert/layer to answer CE.

### Why it may work better

Текущая importance формула `-dot(g, w_i*h_i)` является локальной линейной аппроксимацией. Delta CE проверяет фактический forward effect от intervention, поэтому может лучше разделить experts, которые реально меняют answer quality.

### How to implement in current project

Не менять основную evaluation. Добавить offline scoring mode рядом с `compute_expert_importance`:

1. Получить baseline `abcd_logits` и `L_base`.
2. Для candidate experts из текущего top-k сделать временную intervention:
   - либо занулить routing weight expert `e` в custom forward hook;
   - либо добавить временный negative bias к `layer.mlp.gate.bias[e]`;
   - либо, проще для первого эксперимента, оценивать layer/group delta, а не every expert.
3. Повторить forward на тех же `conditioning_ids`.
4. Считать `delta_ce`.

Практичный первый вариант: не делать per-expert для всех экспертов, а брать top candidates из gradient importance и перепроверять только их.

### Required tensors / data

- `conditioning_ids`;
- `abcd_logits`, `correct_idx`;
- candidate layer/expert ids;
- ability to temporarily patch router bias or MLP forward;
- baseline CE and modified CE.

### Expected effect

Снизить false positives/false negatives при expert selection. Особенно полезно для negative experts: если suppression эксперта улучшает CE, это сильный сигнал.

### Risks / caveats

- Дорого: per-expert interventions требуют много extra forward passes.
- Temporary router bias может менять не только выбранного expert, а весь top-k composition.
- Патч MLP forward для точной expert ablation зависит от `trust_remote_code` Qwen MoE internals и требует проверки.

### Experiment plan

1. Offline на 20-30 calibration samples.
2. Проверять только top-3 positive и bottom-3 negative experts per selected layer.
3. Сравнить signs с gradient importance.
4. Если agreement низкий, использовать delta CE как фильтр, а не полный replacement.

### Priority

Medium.

## 8. Loss proposal 4: Delta margin causal effect

### Idea

Измерять causal effect не по CE, а по answer margin. Это ближе к exact-choice метрике и дешевле интерпретировать.

### Formula

Baseline margin:

```math
m_{base} = z_y^{base} - \max_{c \ne y} z_c^{base}
```

Modified/intervened margin:

```math
m_{mod} = z_y^{mod} - \max_{c \ne y} z_c^{mod}
```

Delta margin:

```math
\Delta m_e = m_{mod} - m_{base}
```

Если intervention - suppression/downweight expert:

```text
Delta m_e < 0: suppression ухудшил margin, expert positive.
Delta m_e > 0: suppression улучшил margin, expert negative.
```

### What signal it captures

Фактический effect expert/layer на decision boundary gold-vs-best-wrong.

### Why it may work better

Многие результаты в таблице отличаются на 0.01-0.03 accuracy. Для таких задач важны boundary cases, где ближайший wrong answer почти равен gold. Delta margin фокусируется именно на этом.

### How to implement in current project

Почти как Delta CE, но score:

```python
def answer_margin(abcd_logits, correct_idx):
    gold = abcd_logits[correct_idx]
    wrong = torch.cat([abcd_logits[:correct_idx], abcd_logits[correct_idx + 1:]])
    return gold - wrong.max()
```

Добавить function for causal rescore:

```python
delta_margin = margin_modified - margin_base
```

Использовать как sign filter для `selected_pos` / `selected_neg` в `apply_expert_bias_v6`.

### Required tensors / data

- baseline and modified `abcd_logits`;
- `correct_idx`;
- candidate expert/layer ids;
- temporary intervention mechanism.

### Expected effect

Лучшее соответствие selected experts фактическому изменению answer rank.

### Risks / caveats

- Те же compute-cost risks, что у Delta CE.
- Margin может игнорировать изменения confidence, если rank не близок к boundary.
- Требует аккуратного выбора intervention magnitude.

### Experiment plan

1. Использовать на subset candidate layers after Otsu.
2. Фильтровать experts, у которых sign gradient importance и sign delta margin конфликтуют.
3. Сравнить evaluation с текущим `v6` на physics и chemistry.

### Priority

High for validation/filtering, Medium as full scoring replacement.

## 9. Loss proposal 5: Answer-only KL to baseline

### Idea

Использовать baseline answer distribution как teacher, чтобы intervention не разрушал уже полезное поведение, но добавить supervised tilt к gold answer.

### Formula

Baseline answer distribution:

```math
p_0 = softmax(z^{base}_{ABCD}/T)
```

Current/modified distribution:

```math
p = softmax(z_{ABCD}/T)
```

KL preservation:

```math
L_{KL-ABCD} = T^2 \cdot KL(p_0 || p)
```

Gold-boosted teacher:

```math
q = normalize((1-\alpha)p_0 + \alpha onehot(y))
```

```math
L = KL(q || p)
```

### What signal it captures

Сохраняет answer distribution baseline, но усиливает gold. Это полезно, если bias должен быть небольшим и не ломать unrelated behavior.

### Why it may work better

Текущий CE может агрессивно толкать модель к gold даже на noisy/ambiguous examples. KL teacher сохраняет относительную структуру answer probabilities и может уменьшить regressions на соседних datasets.

### How to implement in current project

Для pure calibration без intervention baseline equals current, поэтому KL полезен в двух режимах:

1. как post-bias regularizer при подборе bias;
2. как offline scoring для candidate intervention.

В `compute_expert_importance` можно сохранить baseline `abcd_logits.detach()` per sample. Затем в modified forward сравнивать с ним.

### Required tensors / data

- baseline `abcd_logits_base`;
- modified/current `abcd_logits`;
- `correct_idx`;
- temperature `T`;
- gold mixing coefficient `alpha`.

### Expected effect

Меньше destructive interventions; selected layers/expert sets должны быть более conservative.

### Risks / caveats

- Если baseline ошибается уверенно, KL preservation может удерживать неправильное поведение.
- Answer-only KL не видит full-vocab format distribution.
- Требует baseline/modified comparison, которого сейчас нет.

### Experiment plan

1. Не использовать как primary gradient loss.
2. Использовать как penalty при выборе candidate bias sets:

   ```text
   score = delta_gold_margin - gamma * KL_ABCD
   ```
3. Проверять на unrelated domain degradation.

### Priority

Medium.

## 10. Loss proposal 6: Top-N vocab KL to baseline

### Idea

Расширить KL preservation за пределы `A/B/C/D`, но не считать full-vocab KL по всему словарю. Использовать top-N baseline vocab logits на answer position.

### Formula

Пусть `S` - union множества:

```text
top_N_vocab_tokens_from_baseline ∪ token_ids(A/B/C/D)
```

```math
p_0^S = softmax(z_{base}^S/T)
```

```math
p^S = softmax(z_{mod}^S/T)
```

```math
L_{topN-KL} = T^2 \cdot KL(p_0^S || p^S)
```

Combined:

```math
L = L_{answer} + \gamma L_{topN-KL}
```

### What signal it captures

Сохраняет local language-model distribution на answer position, включая форматные tokens, punctuation и common next tokens, но дешевле full-vocab KL.

### Why it may work better

CE over four letters слишком узкий. Если intervention повышает `A/B/C/D`, но ломает формат или next-token distribution, answer-only loss этого не видит. Top-N KL может penalize слишком сильные router shifts.

### How to implement in current project

В `compute_expert_importance` уже есть `logits_pos`. Для baseline:

```python
top_vals, top_ids = torch.topk(logits_pos.detach(), N)
answer_ids = ...
support_ids = torch.unique(torch.cat([top_ids, answer_token_ids]))
```

Для modified comparison нужен второй forward после intervention. Для gradient-only текущего pass можно использовать `L = CE + gamma * entropy/topN regularizer`, но полноценный KL требует baseline snapshot.

### Required tensors / data

- full vocab `logits_pos`;
- baseline full vocab logits at `pred_pos`;
- modified logits;
- answer token ids;
- `N`, `T`, `gamma`.

### Expected effect

Более безопасный bias selection, меньше неожиданных regressions из-за слишком сильного routing shift.

### Risks / caveats

- Дополнительная память для хранения top-N support.
- Если `N` слишком мал, KL пропустит важные alternatives; если слишком велик, будет дорого.
- Может penalize нужные changes.

### Experiment plan

1. Начать с `N=32` или `N=64`, `T=2`, `gamma=0.05`.
2. Использовать только как regularizer in candidate scoring.
3. Сравнить selected layers/expert overlap и unrelated degradation.

### Priority

Medium.

## 11. Loss proposal 7: Router distribution stability loss

### Idea

Добавить loss/score, который контролирует, насколько сильно proposed bias меняет router distribution. Для проекта это важно, потому что intervention применяется именно к router logits.

### Formula

Router probabilities per layer:

```math
r_l = softmax(g_l)
```

Stability KL:

```math
L_{router-KL,l} = KL(r_l^{base} || r_l^{mod})
```

Top-k overlap:

```math
overlap_l = \frac{|TopK(r_l^{base}) \cap TopK(r_l^{mod})|}{K}
```

Instability penalty:

```math
P_{unstable,l} = 1 - overlap_l
```

Selected expert reward/penalty:

```math
R_l =
mean_{e \in E_l^+} \Delta r_{l,e}
- mean_{e \in E_l^-} \Delta r_{l,e}
- \gamma KL(r_l^{base} || r_l^{mod})
```

### What signal it captures

Насколько intervention реально двигает routing в нужную сторону и насколько сильно ломает остальной routing.

### Why it may work better

Текущий layer score учитывает gap, но не измеряет actual post-bias router distribution. Router KL / overlap помогает отличать:

- bias слишком слабый: selected expert probability почти не изменилась;
- bias слишком сильный: top-k полностью другой, risk high;
- bias полезный: positive experts выросли, negative experts снизились, top-k mostly stable.

### How to implement in current project

После candidate `delta_bias` в `apply_expert_bias_v6` можно offline посчитать expected router distribution shift на сохранённых `all_router_logits[layer_idx]`:

```python
base_logits = torch.stack(all_router_logits[layer_idx]).float()
mod_logits = base_logits + delta_bias
base_probs = F.softmax(base_logits, dim=-1)
mod_probs = F.softmax(mod_logits, dim=-1)
router_kl = ...
topk_overlap = ...
```

Это не требует повторного model forward.

### Required tensors / data

- `all_router_logits[layer_idx]`;
- candidate `delta_bias`;
- selected positive/negative expert ids;
- `top_k`.

### Expected effect

Снизить over-biasing и улучшить выбор `eps`/layers. Это дешёвый offline score.

### Risks / caveats

- Router logits сохранены только на answer token positions из calibration.
- Softmax shift по сохранённым logits не учитывает downstream hidden-state changes после modified routing.
- Слишком сильная stability penalty может запретить полезные routing changes.

### Experiment plan

1. Добавить logging `router_kl`, `topk_overlap`, positive/negative probability delta.
2. Не менять bias сначала, только сравнить текущие experiments.
3. Затем добавить rejection rule: skip layer if `topk_overlap < 0.5` или `router_kl > threshold`.

### Priority

High.

## 12. Loss proposal 8: Expert contrastive ranking loss

### Idea

Явно разделять positive и negative experts через ranking objective вместо независимых thresholds.

### Formula

Пусть `s_e` - expert score. Возможные варианты:

- gradient importance `I_e`;
- router probability delta `Delta r_e`;
- causal delta margin `Delta m_e`;
- causal delta CE `-Delta L_e`.

Ranking loss:

```math
L_{rank} =
\frac{1}{|P||N|}
\sum_{p \in P}
\sum_{n \in N}
\max(0, \tau - s_p + s_n)
```

Score-only selection objective:

```math
Score_l = mean_{p \in P_l} s_p - mean_{n \in N_l} s_n
```

### What signal it captures

Не просто "какие experts positive", а насколько reliably positive experts сильнее negative experts в одном слое.

### Why it may work better

Текущий `v6` выбирает positive по `mean(positive)` и negative по `mean(abs(negative))+std`. Это local thresholds, не учитывающие separation между группами. Contrastive ranking лучше подходит для выбора слоёв, где есть чистое разделение helpful/harmful experts.

### How to implement in current project

На первом этапе не нужен backward по `L_rank`. Использовать ranking score как layer/expert selection metric в `apply_expert_bias_v6`:

```python
pos_scores = importances[selected_pos]
neg_scores = importances[selected_neg].abs()
separation = pos_scores.mean() - neg_scores.mean()
```

Более корректно:

```python
pairwise_margin = (pos_scores[:, None] - neg_scores[None, :]).mean()
```

Для causal version заменить `importances` на `delta_margin` scores.

### Required tensors / data

- normalized expert importances;
- selected positive/negative expert ids;
- optionally causal delta scores;
- layer grouping.

### Expected effect

Выбирать меньше слоёв с mixed/noisy signs и больше слоёв, где experts clearly separated.

### Risks / caveats

- Если negative experts мало или они отсутствуют, ranking score плохо определён.
- Может отбрасывать positive-only useful layers.
- Causal version дорогая.

### Experiment plan

1. Использовать as additional layer_score term:

   ```text
   layer_score = old_layer_score * sigmoid(separation / sigma)
   ```
2. Сравнить Otsu-selected layers до/после.
3. Запустить small evaluation на domains with observed mixed results.

### Priority

Medium.

## 13. Loss proposal 9: Robust layer score

### Idea

Не менять token-level loss, а улучшить агрегирование expert/layer signal. Текущий score:

```text
layer_score = (eps * mean(selected_importance)) / gap
```

может быть чувствителен к выбросам на 100 calibration samples.

### Formula

Robust positive score:

```math
S_l^+ =
\frac{\epsilon_l \cdot robust\_mean(I_{l,E^+})}
{gap_l + \eta}
\cdot stability_l
```

Robust mean variants:

```math
robust\_mean(x) = median(x)
```

или trimmed mean:

```math
trimmed\_mean_\alpha(x) =
mean(x_{\alpha ... 1-\alpha})
```

Signal-to-noise:

```math
SNR_l^+ = \frac{mean(I_{l,E^+})}{std(I_{l,E^+}) + \eta}
```

Final:

```math
S_l^+ =
\frac{\epsilon_l \cdot trimmed\_mean(I_{l,E^+})}
{gap_l + \eta}
\cdot \frac{|mean(I_{l,E^+})|}{std(I_{l,E^+})+\eta}
```

Analogously for negative:

```math
S_l^- =
\frac{\epsilon_l \cdot trimmed\_mean(|I_{l,E^-}|)}
{gap_l + \eta}
\cdot SNR_l^-
```

### What signal it captures

Layer usefulness adjusted by effect size, router sensitivity and within-layer stability.

### Why it may work better

В текущем `v6` слой может пройти Otsu из-за нескольких extreme experts. Robust score снижает влияние выбросов и повышает слои с согласованным expert signal.

### How to implement in current project

В `apply_expert_bias_v6` заменить:

```python
pos_mean = importances[selected_pos].mean().item()
neg_mean_abs = importances[selected_neg].abs().mean().item()
pos_auto_score = (eps * pos_mean) / safe_gap
neg_auto_score = (eps * neg_mean_abs) / safe_gap
```

на helper:

```python
pos_score = robust_layer_score(importances[selected_pos], eps, safe_gap)
neg_score = robust_layer_score(importances[selected_neg].abs(), eps, safe_gap)
```

### Required tensors / data

- selected positive/negative importances;
- `eps`;
- `safe_gap`;
- optionally per-sample importances if code is extended beyond accumulated sums.

### Expected effect

Более стабильный layer selection; меньше случайных layers от Otsu.

### Risks / caveats

- Если selected experts всего 1-2, median/trimmed mean мало отличается от mean.
- SNR по aggregated expert vector не равен stability across samples. Лучший вариант требует хранить per-sample importances.

### Experiment plan

1. Cheap offline: пересчитать layer scores из existing `expert_importance`/`all_router_logits`.
2. Сравнить selected layers с текущим Otsu.
3. Проверить top selected layers на math/physics.

### Priority

High.

## 14. Loss proposal 10: Calibration stability / bootstrap loss

### Idea

Так как calibration set маленький, оценивать не только mean importance, но и bootstrap stability выбора experts/layers.

### Formula

Для bootstrap split `b`:

```math
I_{l,e}^{(b)} = \sum_{i \in B_b} importance_{i,l,e}
```

Stability sign score:

```math
stab_{l,e} =
\left| mean_b(sign(I_{l,e}^{(b)})) \right|
```

Bootstrap effect:

```math
\mu_{l,e} = mean_b(I_{l,e}^{(b)})
```

```math
\sigma_{l,e} = std_b(I_{l,e}^{(b)})
```

```math
score_{l,e} = \frac{\mu_{l,e}}{\sigma_{l,e} + \eta} \cdot stab_{l,e}
```

Layer score:

```math
S_l = mean_{e \in selected} |score_{l,e}|
```

### What signal it captures

Насколько expert/layer signal устойчив к выбору calibration samples.

### Why it may work better

100 samples - мало. Если expert importance держится только на нескольких examples, такой expert может плохо переноситься на full evaluation. Bootstrap stability явно penalize нестабильные experts.

### How to implement in current project

Нужно изменить `compute_expert_importance`, чтобы сохранять per-example importance:

```python
per_sample_importance[i][layer_idx][expert_idx] = importance
```

Затем добавить offline aggregation:

```python
bootstrap_scores = compute_bootstrap_scores(per_sample_importance, n_boot=50)
```

Для первого эксперимента можно использовать deterministic splits вместо bootstrap: 5 folds по 20 samples.

### Required tensors / data

- per-sample expert importance, сейчас не сохраняется;
- layer/expert ids;
- bootstrap/fold config.

### Expected effect

Меньше overfitting calibration set; selected experts должны быть стабильнее across calibration subsets.

### Risks / caveats

- Требует больше памяти и structured storage.
- На 100 samples bootstrap variance может быть грубой.
- Если domain signal реально sparse, stability penalty может откинуть полезные rare experts.

### Experiment plan

1. Добавить per-sample importance logging без изменения bias.
2. Сравнить sign agreement across 5 folds.
3. Использовать stability как multiplier для current importances:

   ```text
   adjusted_importance = importance * stability
   ```
4. Запустить small evaluation.

### Priority

High.

## 15. Multi-objective scoring function

Лучший практический вариант, вероятно, не один loss, а scoring function для selection:

```math
FinalScore =
\alpha \cdot AnswerQualityScore
+ \beta \cdot RouterEffectScore
- \gamma \cdot RoutingInstabilityPenalty
- \delta \cdot VariancePenalty
+ \zeta \cdot BootstrapStability
```

Практичная версия для текущего проекта:

```math
S_l^+ =
\alpha \cdot \frac{\epsilon_l \cdot robust\_mean(I_{l,E^+})}{gap_l+\eta}
- \gamma \cdot KL(r_l^{base} || r_l^{bias})
- \delta \cdot Var(I_{l,E^+})
+ \zeta \cdot Stability_l^+
```

```math
S_l^- =
\alpha \cdot \frac{\epsilon_l \cdot robust\_mean(|I_{l,E^-}|)}{gap_l+\eta}
- \gamma \cdot KL(r_l^{base} || r_l^{bias})
- \delta \cdot Var(|I_{l,E^-}|)
+ \zeta \cdot Stability_l^-
```

Suggested defaults:

| Coefficient |  Default | Meaning                                                                                          |
| ----------- | -------: | ------------------------------------------------------------------------------------------------ |
| `alpha`   |  `1.0` | Основной answer/expert importance signal.                                                |
| `beta`    | `0.25` | Reward за рост positive expert router probability / снижение negative probability. |
| `gamma`   | `0.05` | Penalty за router KL.                                                                          |
| `delta`   | `0.10` | Penalty за variance/noise.                                                                     |
| `zeta`    | `0.25` | Reward за bootstrap/fold stability.                                                            |

Практический порядок внедрения:

1. Сначала добавить robust layer score и router stability logging.
2. Затем добавить bootstrap stability.
3. Только после этого пробовать causal delta CE/margin как expensive validation filter.

## 16. Recommended experiment plan

### Stage 1: cheap offline scoring

Без полного запуска всех датасетов:

- добавить `--loss_type` и посчитать scores на calibration set;
- для каждого loss сохранить или напечатать:
  - selected positive layers;
  - selected negative layers;
  - selected positive/negative experts per layer;
  - distribution of importances;
  - Otsu thresholds;
  - router KL / top-k overlap if available;
  - overlap с текущим CE/v6 selection.
- сравнить:
  - CE vs answer margin;
  - current mean layer score vs robust layer score;
  - current selection vs stability-adjusted selection.

Минимальные домены для Stage 1:

- `math`;
- `physics`;
- `chemistry`.

### Stage 2: small evaluation

Для каждого перспективного loss:

- запустить calibration dataset;
- запустить 1-2 соседних датасета из той же области;
- запустить 1 unrelated domain для проверки деградации.

Пример:

```text
physics calibration:
- college_physics*
- conceptual_physics
- high_school_physics
- abstract_algebra or high_school_biology as unrelated sanity check
```

Сравнивать не только accuracy, но и:

- change vs baseline;
- change vs current `v6`;
- selected layer count;
- negative layer count;
- router stability.

### Stage 3: full evaluation

Запустить на всех доменах из таблицы:

- Mathematics;
- Physics;
- History;
- Biology;
- Chemistry.

Рекомендуемый shortlist для full evaluation:

1. Answer margin loss.
2. Robust layer score with current CE.
3. Current CE + router stability penalty.
4. Bootstrap/stability-adjusted importance.
5. Delta margin filter, если Stage 1/2 покажет разумную стоимость.

## 17. Final recommendation

| Rank | Loss proposal                                   | Priority    | Why                                                                                                                                            | Implementation cost | Expected risk                                                                                                 |
| ---- | ----------------------------------------------- | ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- | ------------------------------------------------------------------------------------------------------------- |
| 1    | Loss 2: Answer margin loss                      | High        | Близко к exact-choice evaluation, использует уже доступные `abcd_logits`, дешёвый replacement для CE. | Low                 | Может дать слабый gradient на easy examples; лучше начинать с softplus margin. |
| 2    | Loss 9: Robust layer score                      | High        | Бьёт прямо в текущую слабую точку `layer_score = mean/gap`, снижает noise от 100-sample calibration.    | Low-Medium          | Может скрыть rare but useful experts.                                                              |
| 3    | Loss 7: Router distribution stability loss      | High        | Совместим с routing-bias задачей; можно считать offline из `all_router_logits` и candidate `delta_bias`.   | Medium              | Слишком сильная penalty может запретить полезные routing changes.         |
| 4    | Loss 10: Calibration stability / bootstrap loss | High        | Наиболее релевантно малому calibration set; улучшает устойчивость selected experts/layers.         | Medium              | Нужно хранить per-sample importance.                                                              |
| 5    | Loss 4: Delta margin causal effect              | Medium-High | Проверяет фактическое влияние experts на answer boundary; хороший фильтр false positives.            | High                | Дорогой из-за дополнительных forward/interventions.                                  |

Не рекомендую начинать с full vocab KL как первого шага: он концептуально полезен, но дороже и менее напрямую связан с текущей selection pipeline. Top-N KL можно добавить позже как safety regularizer.

## 18. Required code changes

Минимальные изменения для loss experiments:

1. В `main.py` добавить CLI:

```text
--loss_type ce | ce_refined | margin | soft_margin
--answer_margin_tau
--temperature
--label_smoothing
--position_decay
```

2. Вынести loss из `compute_expert_importance` в helper:

```python
compute_answer_loss(abcd_logits, correct_idx, loss_type, **kwargs)
```

3. Для robust layer score добавить helper в `apply_expert_bias_v6`:

```python
robust_layer_score(values, eps, gap, mode="trimmed_mean")
```

4. Для router stability добавить offline computation внутри `apply_expert_bias_v6` перед применением bias:

```python
estimate_router_shift(all_router_logits[layer_idx], delta_bias, top_k)
```

5. Для bootstrap stability изменить return format `compute_expert_importance`:

```text
expert_importance
all_router_logits
per_sample_importance
```

6. Для delta CE / delta margin добавить отдельный expensive validation mode:

```text
--causal_filter none | delta_ce | delta_margin
--causal_filter_top_n_experts
--causal_intervention_strength
```

Важно: исходная evaluation pipeline (`generate_responses`, `test_accuracy`) может остаться без изменений.

## 19. Risks and caveats

- Любой новый loss меняет gradient, а значит меняет signs expert importance. Старые experiment results нельзя напрямую сравнивать без повторного baseline/current-v6 run.
- Слишком сложный objective может overfit 100 calibration samples.
- Causal/delta losses могут быть вычислительно дорогими и требовать аккуратного patching Qwen MoE internals.
- Router KL/top-k overlap, посчитанные на saved router logits, являются approximation: после real modified routing hidden states downstream могут измениться.
- Answer-only losses не видят full-vocab format failures.
- Full/top-N vocab KL может сохранять baseline mistakes.
- Если extraction regex ловит случайную букву, любой answer-position loss будет шумным. Нужно логировать и, возможно, reweight по `extracted_gen_idx` и confidence.
- Baseline branch и calibration sample size имеют caveats, уже описанные в `project_description.md`; перед крупными запусками стоит зафиксировать protocol.

## 20. Unclear or not fully recoverable from code

- Точная внутренняя реализация Qwen MoE forward не видна в репозитории, потому что модель загружается через `trust_remote_code=True` из external model path. Поэтому exact expert ablation hooks требуют ручной проверки на этой модели.
- Старые эксперименты с `top-5`, `top-3`, `top-1`, quantile thresholds и old loss не полностью восстановимы по текущему `main.py`; часть логики осталась только в комментариях и experiment table.
- В текущем коде нет structured storage для per-sample importance, поэтому bootstrap proposal требует изменения формата данных.
- Неясно, насколько дорого будет repeated causal intervention на доступном GPU setup; это нужно измерить на 10-20 samples.
- `project_description.md` при чтении в текущей Windows console отображался с encoding mojibake, но сам файл существует и структура была сверена по содержанию и коду. Для дальнейших правок лучше открывать markdown в IDE с UTF-8.
