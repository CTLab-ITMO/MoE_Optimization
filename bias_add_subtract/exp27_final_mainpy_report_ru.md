# exp27: финальный отчет по воспроизводимому main.py scratch pipeline

Каноническая команда лучшего результата:

```bash
python main.py --method exp27_best --domains all --recompute-from-scratch --output-dir results/exp27_scratch_repro_20260618_002 --seed 1235
```

Канонический run: `results/exp27_scratch_repro_20260618_002`.

Главное: этот результат получен полным пересчетом с нуля через `main.py`. Старые precomputed artifacts `exp24`-`exp27` не используются как вход для вычисления результата. Старый `exp27_calibration_primary_positive_margin_aux.json` загружается только после завершения scratch-run как reference для сравнения `old exp27 vs scratch exp27`.

## 1. Краткое резюме

`exp27_best` решает задачу безопасного добавления bias к router gate в MoE-модели Qwen. Модель не дообучается, веса трансформера не меняются, optimizer не используется. Меняется только небольшой inference-time сдвиг для выбранных экспертов: некоторые эксперты начинают выбираться router-ом чуть чаще или чуть реже.

Метод сначала прогоняет baseline, затем на calibration dataset ищет экспертов, которые помогают правильным ответам A/B/C/D. После этого он строит candidate pool, выбирает prefix длины `k`, считает масштаб `alpha`, получает итоговые `final_delta` и сохраняет новый calibration artifact. На eval dataset labels не используются для выбора экспертов, `k`, `alpha`, gate или apply/reject. Eval labels используются только после того, как final predictions уже выбраны, чтобы посчитать accuracy для отчета.

Безопасность метода строится на fallback: если label-free gate считает scaling небезопасным для eval dataset, итогом остается baseline prediction. Поэтому метод не доказывает математически отсутствие регрессий, но снижает риск: unsafe scaling не применяется, а baseline сохраняется как запасной вариант.

## 2. Что именно считается финальным методом

Финальный метод - это связка:

| Компонент | Значение                                                            |
| ------------------ | --------------------------------------------------------------------------- |
| Entry point        | `main.py`                                                                 |
| Method             | `exp27_best`                                                              |
| Run mode           | `scratch`, включается флагом `--recompute-from-scratch` |
| Config             | `configs/exp27_scratch_config.json`                                       |
| Canonical output   | `results/exp27_scratch_repro_20260618_002`                                |
| Seed               | `1235`                                                                    |
| Model path         | `/mnt/storage00/usr/ishukin/models/qwen-moe-chat`                         |
| Result JSON        | `results/exp27_scratch_repro_20260618_002/result.json`                    |
| Log                | `results/exp27_scratch_repro_20260618_002/run.log`                        |

Финальный метод не является replay старого JSON. В scratch mode `main.py` заново считает:

1. baseline;
2. calibration;
3. expert importance;
4. candidate experts;
5. raw deltas;
6. `alpha`;
7. выбранный `k`;
8. final deltas;
9. gate reference stats;
10. eval metrics;
11. final predictions;
12. итоговые таблицы.

## 3. Как воспроизвести лучший результат

Запуск:

```bash
cd /mnt/storage00/usr/ishukin/WES

python main.py \
  --method exp27_best \
  --domains all \
  --recompute-from-scratch \
  --output-dir results/exp27_scratch_repro_20260618_002 \
  --seed 1235
```

Config:

```text
configs/exp27_scratch_config.json
```

Важные параметры из config:

| Параметр            | Значение                                               |
| --------------------------- | -------------------------------------------------------------- |
| `method`                  | `exp27_best`                                                 |
| `run_mode`                | `scratch`                                                    |
| `seed`                    | `1235`                                                       |
| `model_path`              | `/mnt/storage00/usr/ishukin/models/qwen-moe-chat`            |
| `candidate_policy`        | `calibration_primary_positive_margin_auxiliary_from_scratch` |
| `loss_top_k`              | `top_k`                                                      |
| `loss_position`           | `canonical_answer_slot`                                      |
| `loss_type`               | `boundary_weighted_margin`                                   |
| `margin_tau`              | `0.0`                                                        |
| `lambda_ce`               | `0.5`                                                        |
| `boundary_scale`          | `1.0`                                                        |
| `boundary_weight_min`     | `0.1`                                                        |
| `boundary_weight_max`     | `1.0`                                                        |
| `importance_type`         | `router_bias_contrast_margin`                                |
| `router_reach_temp`       | `1.0`                                                        |
| `router_reach_bias`       | `0.0`                                                        |
| `v6_eps_policy`           | `top3_boundary_fd`                                           |
| `direct_delta_policy`     | `multiplier`                                                 |
| `direct_delta_multiplier` | `1.0`                                                        |

Output files:

```text
results/exp27_scratch_repro_20260618_002/run.log
results/exp27_scratch_repro_20260618_002/result.json
results/exp27_scratch_repro_20260618_002/calibration_artifacts/
results/exp27_scratch_repro_20260618_002/tables/
results/exp27_scratch_repro_20260618_002/exp27_scratch_reproducibility_report_ru.md
```

В `run.log` есть строка:

```text
SCRATCH MODE: no precomputed artifacts loaded
```

В `result.json` есть audit-поля:

```json
{
  "run_mode": "scratch",
  "precomputed_artifacts_loaded": false,
  "loaded_input_files": [
    "configs/exp27_scratch_config.json",
    "RULES.md"
  ],
  "reference_files_loaded_after_run": [
    "exp27_calibration_primary_positive_margin_aux.json"
  ]
}
```

Это означает: старый `exp27` artifact не участвовал в вычислении, а был открыт только после завершения run для сверки.

## 4. Что является входом, а что нет

Разрешенные входы canonical scratch-run:

| Вход                                                         | Зачем нужен                                                                  |
| ---------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| Модель `/mnt/storage00/usr/ishukin/models/qwen-moe-chat` | исходная MoE-модель                                                      |
| MMLU-like datasets                                               | вопросы, варианты A/B/C/D и labels                                     |
| `main.py`                                                      | точка входа                                                                  |
| `exp27_scratch_core.py`                                        | реализация scratch pipeline                                                  |
| `configs/exp27_scratch_config.json`                            | методологические настройки                                    |
| `RULES.md`                                                     | правила исследования                                                |
| `seed=1235`                                                    | воспроизводимость порядка/рандома, где он есть |

Запрещенные входы в scratch mode:

```text
exp27_calibration_primary_positive_margin_aux.json
exp26_composite_positive_margin_gate.json
exp25_math_physics_prefix_diagnostic.json
exp24_top7_tolerant_gate_artifact.json
любые старые result JSONs exp24-exp27
старые selected experts
старые raw deltas
старые alpha
старые final deltas
старые gate decisions
```

Эти файлы можно читать только после scratch-run, чтобы сравнить новый результат со старым.

## 5. Постановка задачи

Baseline - это обычный запуск исходной модели без добавления expert bias. Для каждого вопроса модель получает prompt и выбирает один из вариантов `A`, `B`, `C`, `D`.

Baseline prediction - выбранный вариант без scaling. Baseline accuracy - доля правильных ответов baseline:

```text
accuracy = number_of_correct_answers / number_of_total_questions
```

Метод считается успешным внутри домена, если:

1. ни один dataset в домене не хуже baseline;
2. хотя бы один non-calibration dataset лучше baseline;
3. eval labels не используются для выбора scaling, gate или fallback;
4. нет per-eval-dataset tuning;
5. метод воспроизводим и объясним.

Calibration dataset - один dataset внутри домена, где labels разрешены для настройки метода. Non-calibration dataset - остальные datasets домена. Улучшение только на calibration dataset считается слабым результатом, поэтому в финальном критерии требуется non-calibration improvement.

## 6. Полный pipeline от начала до конца

Ниже описан фактический scratch pipeline `exp27_best`.

### Шаг 1. Загрузка config и включение scratch mode

`main.py` получает `--method exp27_best` и `--recompute-from-scratch`. После этого он делегирует run в `exp27_scratch_core.py`. В scratch mode старые artifacts не загружаются. Output directory содержит только заново созданные artifacts и таблицы.

### Шаг 2. Загрузка модели и datasets

Загружается модель `/mnt/storage00/usr/ishukin/models/qwen-moe-chat` и tokenizer. Для каждого домена загружаются calibration dataset и все eval datasets из config.

### Шаг 3. Baseline

Для каждого dataset считается baseline accuracy. Это нужно по двум причинам:

1. baseline является точкой сравнения;
2. fallback использует baseline predictions, если scaling отклонен gate.

### Шаг 4. Calibration

На calibration dataset разрешено использовать labels. Метод считает answer logits для `A/B/C/D`, loss, margins и routing/expert signals. Цель calibration - понять, какие experts полезны для правильного ответа и насколько сильно их можно сдвигать.

### Шаг 5. Expert importance

`compute_expert_importance` прогоняет calibration dataset и оценивает, какие эксперты связаны с улучшением correct-answer signal. В финальном config используется:

```text
importance_type = router_bias_contrast_margin
loss_type = boundary_weighted_margin
```

Смысл простой: особенно важны пограничные вопросы, где правильный вариант почти побеждает, но ему не хватает margin. Такие примеры дают сильный сигнал о том, какой expert может помочь изменить ответ в правильную сторону.

### Шаг 6. Candidate experts

После importance scoring эксперты сортируются. Для каждого candidate role берется prefix фиксированного размера: top-3 или top-7, как задано в config. Важно: сами selected experts не лежат в config. Они заново вычисляются в scratch-run и сохраняются в generated calibration artifacts.

### Шаг 7. Raw deltas

`apply_expert_bias_v6` превращает expert importance в `raw_delta`. `raw_delta` - это первый, еще не окончательный bias-сдвиг для конкретного эксперта. Он показывает направление и относительную силу воздействия.

Знак `raw_delta`:

```text
raw_delta > 0  -> bias эксперта надо увеличить
raw_delta < 0  -> bias эксперта надо уменьшить
```

Но raw delta напрямую не считается безопасным. Его масштабирует `alpha`.

### Шаг 8. Alpha через top3_boundary_fd

Для candidate pool считается общий `alpha` через `top3_boundary_fd`. Идея: временно применить raw deltas на calibration dataset и посмотреть finite-difference effect на answer margins.

Для каждого calibration example есть margin правильного ответа:

```text
gold_margin = logit_correct_answer - max_logit_wrong_answer
```

Если baseline был неправильным, но scaling может сделать margin неотрицательным, это `fixed_wrong`. Если baseline был правильным, но scaling может сделать margin отрицательным, это `broken_correct`.

`alpha` выбирается на calibration так, чтобы получить полезный boundary gain и избегать broken-correct случаев. В canonical run `alpha` заново вычисляется для каждого candidate artifact.

### Шаг 9. Dynamic prefix selection: выбор k

Для каждого candidate pool проверяются prefixes:

```text
k = 1, 2, ..., pool_size
```

`k=3` значит взять первых трех experts из отсортированного pool. `k=6` значит взять первых шесть experts. Произвольные подмножества не перебираются: это снижает риск overfitting и делает правило проще.

Правило выбора `k` на calibration:

```text
1. выбрать prefix с лучшей improving calibration accuracy;
2. при равенстве предпочесть меньше experts;
3. если improvement нет, выбрать non-regressing prefix с самым сильным mean_margin_delta;
4. если все prefixes регрессируют, выбрать fallback k=0.
```

Eval labels в этом выборе не участвуют.

### Шаг 10. Final deltas

Итоговый bias для expert:

```text
final_delta_e = raw_delta_e * alpha * direct_delta_multiplier
```

В canonical config:

```text
direct_delta_multiplier = 1.0
```

Значит фактически:

```text
final_delta_e = raw_delta_e * alpha
```

Эти final deltas добавляются в gate bias выбранных MoE experts.

### Шаг 11. Gate reference stats

Для calibration prefix сохраняются reference stats:

```text
calibration_margin_shift
calibration_turnover_rate
calibration_accepted
calibration_improved
```

Они нужны eval gate. Эти stats создаются в scratch-run и сохраняются в calibration artifact.

### Шаг 12. Eval

На каждом eval dataset:

1. считается baseline;
2. считается scaled candidate;
3. считаются label-free gate stats;
4. gate решает apply/reject;
5. если apply - берутся scaled predictions;
6. если reject - берутся baseline predictions;
7. только после этого labels используются для reporting accuracy.

Псевдокод:

```python
for dataset in domain:
    baseline_predictions = run_baseline(dataset)
    scaled_predictions = run_scaled(dataset, generated_artifact.final_deltas)

    gate_stats = compute_label_free_stats(
        baseline_logits,
        scaled_logits,
        baseline_predictions,
        scaled_predictions,
    )

    if gate_accepts(gate_stats, generated_artifact.reference_stats):
        final_predictions = scaled_predictions
    else:
        final_predictions = baseline_predictions

    accuracy = compute_accuracy(final_predictions, labels)  # only reporting
```

## 7. Формулы и точные определения

### Accuracy

```text
accuracy = number_of_correct_answers / number_of_total_questions
```

### Logits A/B/C/D

Для каждого answer slot берутся logits модели. Для букв `A`, `B`, `C`, `D` tokenizer может иметь несколько single-token вариантов. Код `build_abcd_logits` берет максимум по token ids каждой буквы:

```text
abcd_logit_A = max(logits[token_ids_for_A])
abcd_logit_B = max(logits[token_ids_for_B])
abcd_logit_C = max(logits[token_ids_for_C])
abcd_logit_D = max(logits[token_ids_for_D])
```

### Cross-entropy loss

В коде считается стандартный CE:

```text
CE = -log softmax(abcd_logits)[correct_idx]
```

Но финальный config использует не CE как итоговый loss, а `boundary_weighted_margin`.

### Boundary weighted margin loss

Обозначения:

```text
gold_logit = logit правильного ответа
max_wrong_logit = максимальный logit среди неправильных ответов
margin = gold_logit - max_wrong_logit
wrong_logsumexp = logsumexp(logits неправильных ответов)
```

Soft margin loss:

```text
soft_margin_loss = softplus(tau + wrong_logsumexp - gold_logit)
```

В canonical config:

```text
tau = 0.0
boundary_scale = 1.0
boundary_weight_min = 0.1
boundary_weight_max = 1.0
```

Boundary weight:

```text
boundary_weight = clamp(
    exp(-abs(margin) / boundary_scale),
    boundary_weight_min,
    boundary_weight_max
)
```

Итоговый loss:

```text
loss = boundary_weight * soft_margin_loss
```

Зачем это нужно: пограничные примеры получают больший вес, потому что там небольшое изменение expert routing может реально поменять ответ.

### Top-1 margin на eval

Eval gate не знает правильный ответ. Поэтому он смотрит не `gold_margin`, а margin выбранного моделью ответа:

```text
top1_margin = logit_predicted_answer - logit_second_best_answer
```

Mean margin delta:

```text
mean_margin_delta = mean(top1_margin_scaled - top1_margin_baseline)
```

Если `mean_margin_delta >= 0`, scaling в среднем делает top-1 выбор более уверенным. Это не гарантирует accuracy improvement, но является label-free safety signal.

### Turnover

Turnover - доля вопросов, где prediction поменялся:

```text
turnover = number_of_changed_predictions / number_of_total_questions
```

Большой turnover означает, что scaling сильно меняет поведение модели. Это может быть полезно, но также может ломать правильные ответы.

### Router gate bias

Для expert `e`:

```text
gate_score_e = router_score_e + gate_bias_e
```

Метод добавляет:

```text
gate_bias_e := gate_bias_e + final_delta_e
```

Положительный `final_delta_e` делает expert более вероятным. Отрицательный `final_delta_e` делает expert менее вероятным.

## 8. Как выбираются calibration datasets

Calibration datasets зафиксированы в `configs/exp27_scratch_config.json`.

| Domain    | Calibration dataset           | Почему выбран                                                                                                                                        |
| --------- | ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| math      | `elementary_mathematics`    | Центральный и достаточно представительный math dataset; ранее показал стабильный calibration signal. |
| physics   | `college_physics`           | Строгий physics dataset с хорошей проверкой reasoning; используется как calibration anchor.                               |
| biology   | `high_school_biology`       | Широкий school-level biology dataset; подходит как общий calibration set для biology.                                                  |
| history   | `high_school_world_history` | Самый широкий history calibration anchor по тематике.                                                                                      |
| chemistry | `high_school_chemistry`     | Выбран как основной chemistry calibration dataset; дает достаточно сильный calibration signal.                             |

Эти choices не меняются на основе eval-label outcomes в canonical run.

## 9. Как работает метод по доменам

Таблица ниже фиксирует concrete generated outcome canonical scratch-run. Важно: `selected_k` и `alpha` не были входом config. Они заново получены calibration stage и записаны в generated artifacts.

| Domain    | Candidate policy / pool | Selected k | Primary / auxiliary candidate      |           Alpha | Artifact                                                        |
| --------- | ----------------------- | ---------: | ---------------------------------- | --------------: | --------------------------------------------------------------- |
| math      | top-3                   |          3 | primary `math_top3_primary`      |  0.100000000000 | `calibration_artifacts/math_math_top3_primary.json`           |
| physics   | top-7                   |          4 | primary `physics_top7_primary`   | -1.000000000000 | `calibration_artifacts/physics_physics_top7_primary.json`     |
| physics   | top-3                   |          3 | auxiliary `physics_top3_aux`     | -0.483870967742 | `calibration_artifacts/physics_physics_top3_aux.json`         |
| biology   | top-7                   |          2 | primary `biology_top7_primary`   |  0.381340579710 | `calibration_artifacts/biology_biology_top7_primary.json`     |
| history   | top-7                   |          3 | primary `history_top7_primary`   | -0.025000000000 | `calibration_artifacts/history_history_top7_primary.json`     |
| chemistry | top-7                   |          6 | primary `chemistry_top7_primary` | -0.628837719298 | `calibration_artifacts/chemistry_chemistry_top7_primary.json` |

Full selected experts, raw deltas, final deltas и prefix records находятся в этих generated JSON artifacts. Они являются output canonical scratch-run, а не input.

## 10. Как работает gate

Gate решает, применять scaling на eval dataset или оставить baseline.

Формула из canonical config и `exp27_scratch_core.py`:

```text
if mean_margin_delta >= 0:
    apply scaling
else:
    apply only if:
        mean_margin_delta >= -abs(calibration_margin_shift)
        and, if calibration improved:
            eval_turnover <= calibration_turnover
```

Переменные:

| Переменная         | Значение                                                                                               |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `mean_margin_delta`        | Средний сдвиг top-1 margin на eval dataset после scaling. Считается без labels. |
| `calibration_margin_shift` | Такой же margin shift, но на calibration prefix. Считается во время calibration.    |
| `eval_turnover`            | Доля eval predictions, которые изменились после scaling. Без labels.              |
| `calibration_turnover`     | Turnover на calibration prefix.                                                                              |
| `calibration_improved`     | Улучшилась ли calibration accuracy. Это calibration-only label signal.                          |

Простая интерпретация:

1. Если scaling в среднем не ухудшает уверенность top-1 ответа (`mean_margin_delta >= 0`), gate принимает scaling.
2. Если margin немного ухудшился, gate все еще может принять scaling, но только если ухудшение не сильнее calibration reference.
3. Если calibration accuracy улучшилась, eval turnover не должен быть больше calibration turnover: нельзя менять слишком много ответов сильнее, чем было видно на calibration.
4. Если условия не выполнены, используется baseline fallback.

Positive-margin auxiliary logic:

На некоторых доменах есть primary и auxiliary candidate. В canonical config это есть у physics:

```text
primary: physics_top7_primary
auxiliary: physics_top3_aux
```

Для non-calibration eval dataset, если несколько candidates прошли gate и имеют non-negative `mean_margin_delta`, выбирается candidate с largest positive mean margin shift. Это label-free, потому что выбор использует только logits/predictions/margins/turnover, а не правильные labels.

Риск: positive margin не гарантирует улучшение accuracy. Это proxy-сигнал, а не доказательство.

## 11. Где используются labels, а где нет

### Labels используются

| Stage                     | Dataset                | Для чего используются labels                                                                             |
| ------------------------- | ---------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| Calibration               | Calibration dataset    | Loss, correct-answer margin, expert importance,`fixed_wrong`, `broken_correct`, `alpha`, `k`, calibration accuracy. |
| Final reporting           | Eval datasets          | Только после выбора final predictions, чтобы посчитать final accuracy.                       |
| Old-vs-scratch comparison | После scratch-run | Только для сравнения готового результата со старым reference.                   |

### Labels не используются

| Stage                                | Dataset      | Что выбирается без labels                                    |
| ------------------------------------ | ------------ | ---------------------------------------------------------------------------- |
| Eval gate                            | Eval dataset | Apply/reject scaling.                                                        |
| Eval candidate choice                | Eval dataset | Выбор accepted candidate по `mean_margin_delta` и `change_rate`. |
| Eval fallback                        | Eval dataset | Baseline или scaled predictions.                                          |
| Scratch artifact generation for eval | Eval dataset | Не выбирает experts,`k`, `alpha` или thresholds.            |

Ключевой audit rule:

```text
Eval labels не используются до выбора final_predictions.
```

Если eval labels где-либо использовались бы до выбора final predictions, это было бы leakage и метод нельзя было бы считать label-free на eval.

## 12. Результаты по каждому домену

### Math

| Dataset                 |       Baseline |     exp27_best |           Delta | Calibration |
| ----------------------- | -------------: | -------------: | --------------: | ----------: |
| abstract_algebra        | 0.400000000000 | 0.400000000000 | +0.000000000000 |             |
| college_mathematics     | 0.370000000000 | 0.380000000000 | +0.010000000000 |             |
| elementary_mathematics  | 0.417989417989 | 0.420634920635 | +0.002645502646 |           * |
| high_school_mathematics | 0.355555555556 | 0.355555555556 | +0.000000000000 |             |

### Physics

| Dataset             |       Baseline |     exp27_best |           Delta | Calibration |
| ------------------- | -------------: | -------------: | --------------: | ----------: |
| college_physics     | 0.330000000000 | 0.340000000000 | +0.010000000000 |           * |
| conceptual_physics  | 0.514893617021 | 0.519148936170 | +0.004255319149 |             |
| high_school_physics | 0.384105960265 | 0.384105960265 | +0.000000000000 |             |

### Biology

| Dataset             |       Baseline |     exp27_best |           Delta | Calibration |
| ------------------- | -------------: | -------------: | --------------: | ----------: |
| anatomy             | 0.496296296296 | 0.496296296296 | +0.000000000000 |             |
| college_biology     | 0.597222222222 | 0.604166666667 | +0.006944444444 |             |
| high_school_biology | 0.683870967742 | 0.683870967742 | +0.000000000000 |           * |

### History

| Dataset                      |       Baseline |     exp27_best |           Delta | Calibration |
| ---------------------------- | -------------: | -------------: | --------------: | ----------: |
| high_school_european_history | 0.684848484848 | 0.690909090909 | +0.006060606061 |             |
| high_school_us_history       | 0.696078431373 | 0.696078431373 | +0.000000000000 |             |
| high_school_world_history    | 0.713080168776 | 0.721518987342 | +0.008438818565 |           * |
| prehistory                   | 0.651234567901 | 0.651234567901 | +0.000000000000 |             |

### Chemistry

| Dataset               |       Baseline |     exp27_best |           Delta | Calibration |
| --------------------- | -------------: | -------------: | --------------: | ----------: |
| college_chemistry     | 0.460000000000 | 0.470000000000 | +0.010000000000 |             |
| high_school_chemistry | 0.463054187192 | 0.472906403941 | +0.009852216749 |           * |

## 13. Сводная таблица по доменам

| Domain    | Calibration dataset           | Status | Total gain vs baseline | Non-calibration improved dataset(s) |
| --------- | ----------------------------- | ------ | ---------------------: | ----------------------------------- |
| math      | `elementary_mathematics`    | PASS   |        +0.012645502646 | `college_mathematics`             |
| physics   | `college_physics`           | PASS   |        +0.014255319149 | `conceptual_physics`              |
| biology   | `high_school_biology`       | PASS   |        +0.006944444444 | `college_biology`                 |
| history   | `high_school_world_history` | PASS   |        +0.014499424626 | `high_school_european_history`    |
| chemistry | `high_school_chemistry`     | PASS   |        +0.019852216749 | `college_chemistry`               |

Total gain vs baseline:

```text
+0.068196907614
```

Все пять доменов проходят strict criterion: нет dataset ниже baseline, и в каждом домене есть non-calibration improvement.

## 14. Сравнение с baseline

Главное сравнение - против baseline. `exp27_best` не ухудшает ни один dataset:

```text
min(delta_vs_baseline) = 0.0
```

Положительные улучшения:

| Domain    | Improved datasets                                               |
| --------- | --------------------------------------------------------------- |
| math      | `college_mathematics`, `elementary_mathematics`             |
| physics   | `college_physics`, `conceptual_physics`                     |
| biology   | `college_biology`                                             |
| history   | `high_school_european_history`, `high_school_world_history` |
| chemistry | `college_chemistry`, `high_school_chemistry`                |

Старый `exp27` reference был загружен только после run. Таблица `results/exp27_scratch_repro_20260618_002/tables/old_vs_scratch.md` показывает, что scratch result совпал со старым exp27 по всем datasets.

## 15. Почему этот результат считается лучшим

Этот run считается каноническим по четырем причинам.

1. Он воспроизводится через `main.py`, а не отдельным экспериментальным replay script.
2. Он работает в scratch mode: selected experts, `k`, `alpha`, raw deltas и final deltas пересчитываются заново.
3. Он проходит строгий критерий во всех пяти доменах.
4. Он совпадает со старым artifact-level exp27 reference, но не использует его как input.

По сравнению с baseline:

```text
Total gain vs baseline = +0.068196907614
```

По сравнению с историческим exp17:

```text
Total delta vs exp17 = +0.024255319149
```

Главное улучшение относительно более ранних подходов: теперь есть end-to-end воспроизводимость от raw model + datasets + method config до final result tables.

## 16. Ограничения и риски

1. История метода начиналась с artifact-level scripts. Сейчас canonical result переведен в `main.py` scratch pipeline, но старые scripts остаются историческим фоном.
2. Первый scratch attempt `results/exp27_scratch_repro_20260618_001` был invalid: между candidate runs протекало состояние gate bias. После этого в `exp27_scratch_core.py` был исправлен restore raw baseline gate biases after each candidate. Каноническим считается только `_002`.
3. Calibration labels используются для поиска experts, выбора `k` и `alpha`, поэтому возможен calibration overfit.
4. Gate label-free на eval, но `mean_margin_delta` и turnover - это proxy-сигналы. Они не являются математической гарантией no-regression.
5. Fallback в baseline снижает риск, но может скрывать слабость scaling: если gate часто reject, метод просто не меняет predictions.
6. Результаты имеют небольшие gains, поэтому их желательно проверять bootstrap/rerun stability.
7. Возможна зависимость от seed, порядка данных или недетерминизма GPU. Canonical run фиксирует `seed=1235`, но полная bit-level детерминированность не заявляется.
8. Candidate policy все еще содержит primary/auxiliary roles, а не один абсолютно простой global rule. Это компромисс между качеством и простотой.
9. Positive-margin auxiliary gate может быть оптимистичным: рост top-1 margin не всегда означает рост accuracy.
10. Метод не fine-tuning и не меняет weights, поэтому возможный потолок улучшений ограничен тем, насколько router bias steering способен менять ответы.

## 17. Чеклист для ручной проверки

```text
[ ] Запуск идет через main.py?
[ ] В команде есть --method exp27_best?
[ ] В команде есть --recompute-from-scratch?
[ ] Используется seed 1235?
[ ] Output dir равен results/exp27_scratch_repro_20260618_002?
[ ] В run.log есть SCRATCH MODE: no precomputed artifacts loaded?
[ ] result.json содержит precomputed_artifacts_loaded=false?
[ ] loaded_input_files содержит только config и RULES.md?
[ ] Старый exp27 JSON загружен только после run как reference?
[ ] calibration_artifacts созданы внутри output-dir?
[ ] selected experts/k/alpha/final deltas находятся в generated artifacts?
[ ] Eval gate использует только logits/predictions/margins/turnover?
[ ] Eval labels используются только после выбора final predictions?
[ ] Все calibration datasets отмечены звездочкой в таблицах?
[ ] Есть non-calibration improvement в каждом домене?
[ ] Нет dataset с delta_vs_baseline < 0?
[ ] Scratch result совпадает со старым exp27 reference?
[ ] results/exp27_scratch_repro_20260618_001 не используется как canonical?
```

## 18. Очень короткая версия метода

1. Берем исходную MoE-модель без fine-tuning.
2. Запускаем `main.py --method exp27_best --recompute-from-scratch`.
3. На каждом домене выбираем фиксированный calibration dataset из config.
4. Считаем baseline.
5. На calibration считаем loss/margins и expert importance.
6. Строим candidate pool top-3/top-7 по config.
7. На calibration выбираем `k` и `alpha`.
8. Получаем `final_delta = raw_delta * alpha`.
9. Сохраняем generated calibration artifact.
10. На eval применяем scaling только если label-free gate принимает; иначе оставляем baseline.
11. Accuracy считаем только после фиксации final predictions.
12. Canonical run `_002` проходит все пять доменов без регрессий ниже baseline.
