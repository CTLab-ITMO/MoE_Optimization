# Хронология экспериментов по physics

Примечание по структуре: нумерованные разделы `0-21` идут в хронологическом порядке выполнения экспериментов и фиксации ключевых результатов. Ненумерованные разделы вроде `Общий вывод`, `Что пробовать следующим` и `Приоритет` являются промежуточными аналитическими вставками, которые были актуальны на том этапе исследования; они не являются отдельными экспериментами.

## 0. Бейзлайн из `project_description.md`

| Эксперимент | college_physics | conceptual_physics | high_school_physics | all_part |
| ---------------------- | --------------: | -----------------: | ------------------: | -------: |
| Baseline               |            0.35 |               0.51 |                0.38 |     0.60 |

Это опорная точка. Для physics важно не только поднять `college_physics`, который одновременно используется как calibration/evaluation slice, но и не ломать перенос на `conceptual_physics` и `high_school_physics`.

## 1. Лучший старый physics-результат из таблицы проекта

| Эксперимент                                      | college_physics | conceptual_physics | high_school_physics |
| ----------------------------------------------------------- | --------------: | -----------------: | ------------------: |
| new loss + automatic Otsu layer selection, negative Otsu x2 |            0.35 |               0.52 |                0.38 |

Вывод: старая CE/Otsu-логика не улучшала calibration slice, но давала лучший перенос на `conceptual_physics` и не ухудшала `high_school_physics`. Это важный ориентир: новые методы должны либо поднять хотя бы один physics-набор без регрессий, либо дать более понятный sparse-механизм, который можно развивать.

## 2. Canonical answer-slot + soft margin

Конфиг:

```bash
python main.py --method v6 --mode single --domain physics --loss_top_k top_k --loss_position canonical_answer_slot --loss_type soft_margin --margin_tau 0.0
```

Ключевые calibration-числа:

| Метрика                             | Значение |
| ------------------------------------------ | ---------------: |
| calibration top-1 correct                  |           33/100 |
| mean p_correct                             |           ~0.347 |
| mean margin                                |          ~-0.556 |
| positive-margin samples                    |               33 |
| near-boundary samples,`abs(margin) <= 1` |               12 |
| hard-wrong samples,`margin < -3`         |               33 |

Выбранные слои:

```text
positive: [0, 1, 6, 7, 20, 21, 22, 23]
negative: [2, 4, 5, 19]
```

Результаты:

| Эксперимент | college_physics | conceptual_physics | high_school_physics |
| ---------------------- | --------------: | -----------------: | ------------------: |
| canonical soft-margin  |            0.36 |           0.510638 |            0.364238 |

Вывод: `soft_margin` при `tau=0` фактически совпадает с CE по A/B/C/D, поэтому это не дало нового сигнала. Небольшой плюс на `college_physics` сопровождался ухудшением `high_school_physics`, значит problem не только в позиции loss, но и в том, какие примеры и эксперты формируют gradient.

Следующий шаг после этого: добавить boundary-weighted margin loss, чтобы уменьшить вклад easy-correct и hopeless-wrong примеров и усилить near-boundary случаи.

## 3. Boundary-weighted margin loss, мягкий конфиг

Конфиг:

```bash
python main.py --method v6 --mode single --domain physics --loss_top_k top_k --loss_position canonical_answer_slot --loss_type boundary_weighted_margin --margin_tau 0.0 --boundary_scale 1.0 --boundary_weight_min 0.1 --boundary_weight_max 1.0
```

Ключевые calibration-числа из `boundary_weighted_margin_logs.log`:

| Группа    | count | mean_margin | mean_boundary_weight | mean_weighted_loss |
| --------------- | ----: | ----------: | -------------------: | -----------------: |
| easy_correct    |    28 |    5.924107 |             0.105181 |           0.006171 |
| close_correct   |     5 |    0.443750 |             0.644997 |           0.453342 |
| flippable_wrong |     7 |   -0.562500 |             0.588968 |           0.700376 |
| hard_wrong      |    60 |   -3.661849 |             0.131374 |           0.453314 |

Выбранные слои:

```text
positive: [0, 6, 20, 21, 22, 23]
negative: [1, 2, 4, 5, 19]
```

Результаты:

| Эксперимент                | college_physics | conceptual_physics | high_school_physics |
| ------------------------------------- | --------------: | -----------------: | ------------------: |
| boundary-weighted, scale=1.0, min=0.1 |            0.35 | нет в логе |  нет в логе |

Вывод: идея логически правильная, но в таком виде hard-wrong группа все равно остается слишком влиятельной из-за большого количества примеров. Даже с малым средним весом hard-wrong примеры дают значительную массу loss. Поэтому следующий шаг был естественным: усилить отсечение hard-wrong через меньшие `scale` и `w_min`.

## 4. Boundary-weighted sweep, более строгое отсечение

Конфиг из `hard_wrong_boundary_weight_sweep_logs.log`:

```bash
python main.py --method v6 --mode single --domain physics --loss_top_k top_k --loss_position canonical_answer_slot --loss_type boundary_weighted_margin --margin_tau 0.0 --boundary_scale 0.75 --boundary_weight_min 0.03 --boundary_weight_max 1.0
```

Ключевые calibration-числа:

| Группа    | count | mean_margin | mean_boundary_weight | mean_weighted_loss |
| --------------- | ----: | ----------: | -------------------: | -----------------: |
| easy_correct    |    28 |    5.924107 |             0.034772 |           0.002841 |
| close_correct   |     5 |    0.443750 |             0.558562 |           0.391648 |
| flippable_wrong |     6 |   -0.513021 |             0.531988 |           0.601458 |
| hard_wrong      |    61 |   -3.615907 |             0.062944 |           0.180291 |

Выбранные слои:

```text
positive: [0, 6, 20, 21, 22, 23]
negative: [2, 4, 5, 19, 20, 21]
```

Результаты:

| Эксперимент                  | college_physics | conceptual_physics | high_school_physics |
| --------------------------------------- | --------------: | -----------------: | ------------------: |
| boundary-weighted, scale=0.75, min=0.03 |            0.36 |           0.502128 |            0.357616 |

Сравнение с baseline:

| Dataset             |     Delta |
| ------------------- | --------: |
| college_physics     |     +0.01 |
| conceptual_physics  | -0.007872 |
| high_school_physics | -0.022384 |

Вывод: более сильное boundary-отсечение действительно снижает вклад hard-wrong примеров, но не решает главную проблему. Слой selection становится конфликтным: появляются одновременно positive и negative сигналы вокруг поздних слоев 20-21, а перенос ухудшается. Это говорит, что один только gradient/weighted-loss signal недостаточен: он ранжирует кандидатов, но не проверяет фактический causal effect router-bias интервенции.

Следующий шаг после этого: Delta-margin causal filter как второй этап, который проверяет, меняет ли конкретный expert bias реальный answer margin в нужную сторону.

## 5. Delta-margin causal filter поверх boundary-weighted loss

Конфиг из `causal_filter_delta_margin_logs.log`:

```bash
python main.py --method v6 --mode single --domain physics --loss_top_k top_k --loss_position canonical_answer_slot --loss_type boundary_weighted_margin --margin_tau 0.0 --boundary_scale 1.0 --boundary_weight_min 0.1 --boundary_weight_max 1.0 --causal_filter_delta_margin --causal_filter_samples 32 --causal_filter_min_delta 0.01
```

Ключевые calibration-числа loss до causal filtering совпали с мягким boundary-конфигом:

| Группа    | count | mean_margin | mean_boundary_weight | mean_weighted_loss |
| --------------- | ----: | ----------: | -------------------: | -----------------: |
| easy_correct    |    28 |    5.924107 |             0.105181 |           0.006171 |
| close_correct   |     5 |    0.443750 |             0.644997 |           0.453342 |
| flippable_wrong |     7 |   -0.562500 |             0.588968 |           0.700376 |
| hard_wrong      |    60 |   -3.661849 |             0.131374 |           0.453314 |

Causal filter проверил 14 кандидатов и оставил только 2:

| Layer | Expert | mean_delta_margin | keep |
| ----: | -----: | ----------------: | ---- |
|     2 |     27 |         +0.010254 | yes  |
|     5 |     16 |         +0.027344 | yes  |

Итоговые слои после фильтра:

```text
positive: []
negative: [2, 5]
```

Результаты:

| Эксперимент                         | college_physics | conceptual_physics | high_school_physics |
| ---------------------------------------------- | --------------: | -----------------: | ------------------: |
| boundary-weighted + delta-margin causal filter |            0.37 |           0.506383 |            0.377483 |

Сравнение с baseline:

| Dataset             |     Delta |
| ------------------- | --------: |
| college_physics     |     +0.02 |
| conceptual_physics  | -0.003617 |
| high_school_physics | -0.002517 |

Сравнение со строгим boundary sweep:

| Dataset             |     Delta |
| ------------------- | --------: |
| college_physics     |     +0.01 |
| conceptual_physics  | +0.004255 |
| high_school_physics | +0.019868 |

Вывод: causal filter пока самый полезный новый компонент. Он не просто меняет loss, а проверяет фактическую directionality интервенции. Поэтому он убрал поздние positive-кандидаты и оставил sparse negative intervention в ранних слоях `[2, 5]`. Это логически согласуется с результатами: меньше слоев, меньше переносных регрессий, лучший `college_physics`.

Но это еще не окончательная победа:

- `conceptual_physics` все еще ниже baseline 0.51 и ниже старого CE/Otsu результата 0.52;
- `high_school_physics` почти восстановлен до baseline 0.38, но не превзошел его;
- causal filter использовал только 32 calibration samples, поэтому решение по экспертам может быть шумным;
- критерий keep сейчас смотрит на средний `delta_margin`, а не напрямую на flips/accuracy и не на boundary-weighted delta.

## Общий вывод по текущей серии

Boundary-weighted loss полезен как диагностический и candidate-generation signal, но сам по себе недостаточно надежен. Он правильно показывает, что calibration set состоит в основном из hard-wrong и easy-correct примеров, однако даже строгие веса не гарантируют хороший перенос.

Delta-margin causal filter валиден и полезен: он резко сокращает intervention surface и улучшает профиль относительно обоих boundary-only запусков. Его главный вклад - не в новом loss, а в проверке причинного эффекта конкретного router-bias действия.

На текущих числах лучший новый запуск:

```text
boundary-weighted + delta-margin causal filter:
college_physics      0.37
conceptual_physics   0.506383
high_school_physics  0.377483
```

Он лучше baseline на `college_physics`, почти baseline на `high_school_physics`, но пока чуть хуже на `conceptual_physics`. Значит направление перспективное, но текущий keep-criterion нужно сделать менее шумным и более decision-aware.

## Что пробовать следующим

### 1. Causal filter с большим sample count и более мягким threshold

Почему: в логе есть кандидат Layer 6 Expert 23, у которого `biased_acc` вырос с 0.3125 до 0.34375 на 32 samples, но `mean_delta_margin=0.001953` не прошел порог `0.01`. Это прямой признак, что средний margin может отвергать candidates, которые улучшают discrete answer accuracy.

Рекомендуемый следующий запуск:

```bash
python main.py --method v6 --mode single --domain physics --loss_top_k top_k --loss_position canonical_answer_slot --loss_type boundary_weighted_margin --margin_tau 0.0 --boundary_scale 1.0 --boundary_weight_min 0.1 --boundary_weight_max 1.0 --causal_filter_delta_margin --causal_filter_samples 100 --causal_filter_min_delta 0.0
```

Ожидаемый эффект: проверить, является ли sparse `[2, 5]` устойчивым решением на полном calibration slice, и не были ли отфильтрованы слабые, но accuracy-positive experts.

### 2. Decision-aware causal filter вместо mean-margin-only filter

Что изменить: оставить текущий delta-margin filter, но добавить режим keep по decision statistics:

```text
keep if:
  mean_delta_margin > threshold
  OR correct_flips_gain > 0 with bounded wrong_flips_harm
  OR boundary_weighted_delta_margin > threshold
```

Почему это нужно: глобальная метрика проекта - accuracy после generation/extraction, а не средний margin сам по себе. Если эксперт делает несколько near-boundary flips правильными, но почти не меняет средний margin, текущий фильтр может ошибочно его удалить.

Ожидаемый эффект: сохранить candidates, которые реально меняют ответы, и лучше поднять `conceptual_physics`/`high_school_physics` без расширения intervention surface слишком сильно.

### 3. Format-aware marginal answer loss

Почему: текущий loss все еще слишком завязан на single-token A/B/C/D в canonical slot. Evaluation же работает через generation/extraction, и часть ошибок может быть не в знании ответа, а в surface form. Format-aware aggregation из `proposals_sorted.md` логически следующий шаг после causal filter: он уменьшает tokenization/format mismatch, не превращая задачу в full-vocab KL.

Минимальный вариант:

```text
valid forms per class:
"A", " A", "(A", "A."

class_logit = logsumexp(valid_form_logits)
loss/margin считаются уже по class_logit
```

Ожидаемый эффект: более стабильный answer signal для candidate generation. Особенно полезно, если новые логи покажут, что margin/top-1 на canonical slot плохо коррелирует с generated-answer accuracy.

### 4. Router-constrained selection objective

Почему: boundary loss говорит, какой эксперт полезен для answer objective, но не учитывает, насколько реально и чисто этот эксперт можно сдвинуть router bias. По логам видно, что boundary-only выбор может включать конфликтные поздние слои. Router-constrained score должен штрафовать дорогие/нестабильные router interventions.

Ожидаемый эффект: меньше ложных candidates до causal filter, меньше конфликтов positive/negative в одних и тех же поздних слоях, лучше перенос.

## Приоритет

1. Сначала повторить causal filter на 100 samples и `min_delta=0.0`.
2. Затем реализовать decision-aware causal filter с логами по flips: wrong->correct, correct->wrong, mean_delta на boundary/hard/easy группах.
3. Потом добавлять format-aware marginal answer loss как новый candidate-generation loss.
4. Router-constrained selection имеет смысл после этого, если causal filter останется полезным, но кандидатов будет слишком много или они будут нестабильными.

## 6. Soft Delta-margin causal filter: 100 samples, min_delta=0.0

Конфиг из `soft_causal_filter_logs.log`:

```bash
python main.py --method v6 --mode single --domain physics --loss_top_k top_k --loss_position canonical_answer_slot --loss_type boundary_weighted_margin --margin_tau 0.0 --boundary_scale 1.0 --boundary_weight_min 0.1 --boundary_weight_max 1.0 --causal_filter_delta_margin --causal_filter_samples 100 --causal_filter_min_delta 0.0
```

Ключевые calibration-числа до causal filtering совпадают с мягким boundary-weighted запуском:

| Группа    | count | mean_margin | mean_boundary_weight | mean_weighted_loss |
| --------------- | ----: | ----------: | -------------------: | -----------------: |
| easy_correct    |    28 |    5.924107 |             0.105181 |           0.006171 |
| close_correct   |     5 |    0.443750 |             0.644997 |           0.453342 |
| flippable_wrong |     7 |   -0.562500 |             0.588968 |           0.700376 |
| hard_wrong      |    60 |   -3.661849 |             0.131374 |           0.453314 |

Фильтр проверял candidates на всех 100 calibration samples и с нулевым порогом `min_delta=0.0`. Из-за этого он стал значительно мягче и оставил 9 кандидатов:

| Layer | Expert | Direction | mean_delta_margin | base_acc | biased_acc | improved | worsened | keep |
| ----: | -----: | --------- | ----------------: | -------: | ---------: | -------: | -------: | ---- |
|     0 |     39 | positive  |         +0.004219 |     0.33 |       0.33 |       52 |       43 | yes  |
|     1 |     29 | negative  |         +0.022969 |     0.33 |       0.33 |       54 |       42 | yes  |
|     2 |     27 | negative  |         +0.003672 |     0.33 |       0.33 |       42 |       50 | yes  |
|     6 |      4 | positive  |         +0.014844 |     0.34 |       0.33 |       51 |       47 | yes  |
|     6 |     23 | positive  |         +0.025781 |     0.34 |       0.34 |       54 |       36 | yes  |
|    19 |     44 | negative  |         +0.009844 |     0.34 |       0.34 |       54 |       30 | yes  |
|    20 |     26 | positive  |         +0.000938 |     0.34 |       0.34 |       26 |       24 | yes  |
|    21 |     44 | positive  |         +0.003281 |     0.34 |       0.34 |       19 |       22 | yes  |
|    23 |     21 | positive  |         +0.000156 |     0.34 |       0.34 |       40 |       42 | yes  |

Итоговые слои:

```text
positive: [0, 6, 20, 21, 23]
negative: [1, 2, 19]
```

Результаты:

| Эксперимент                         | college_physics | conceptual_physics | high_school_physics |
| ---------------------------------------------- | --------------: | -----------------: | ------------------: |
| soft causal filter, 100 samples, min_delta=0.0 |            0.35 |           0.502128 |            0.350993 |

Сравнение с baseline:

| Dataset             |     Delta |
| ------------------- | --------: |
| college_physics     |     +0.00 |
| conceptual_physics  | -0.007872 |
| high_school_physics | -0.029007 |

Сравнение с предыдущим causal filter на 32 samples и `min_delta=0.01`:

| Dataset             |     Delta |
| ------------------- | --------: |
| college_physics     |     -0.02 |
| conceptual_physics  | -0.004255 |
| high_school_physics | -0.026490 |

Вывод: этот запуск опроверг гипотезу, что достаточно увеличить sample count и ослабить threshold. Более мягкий causal filter пропускает слишком много слабых candidates с формально положительным `mean_delta_margin`, но без реального улучшения `biased_acc`. Некоторые оставленные эксперты даже имеют `worsened > improved` или ухудшают локальную accuracy:

```text
Layer 2 Expert 27: improved=42, worsened=50, keep=True
Layer 6 Expert 4: base_acc=0.34, biased_acc=0.33, keep=True
Layer 21 Expert 44: improved=19, worsened=22, keep=True
Layer 23 Expert 21: improved=40, worsened=42, keep=True
```

Это показывает, что `mean_delta_margin > 0` сам по себе недостаточен. Он может расти за счет небольших margin-сдвигов, которые не дают правильных flips и даже ухудшают перенос. В итоге intervention surface расширился с `negative: [2, 5]` в лучшем causal-filter запуске до `positive: [0, 6, 20, 21, 23]` и `negative: [1, 2, 19]`, а качество стало хуже baseline на двух внешних physics-наборах.

## Финальное решение по Delta-margin causal filter

Решение: **нет, не продолжать развивать Delta-margin causal filter как следующий основной метод**.

Причина: серия экспериментов показала, что causal filter полезен только как жесткий post-hoc валидатор, но его текущая формула `mean_delta_margin` нестабильна как самостоятельный критерий выбора экспертов. Строгий вариант на 32 samples дал лучший результат серии, но при более полном и мягком запуске метод не подтвердился: он начал пропускать слабые и вредные candidates и ухудшил итоговую accuracy.

Следующее направление должно быть не очередным конфигом Delta-margin causal filter, а новая формула importance, которая сразу лучше отражает суть router-bias интервенции:

```text
boundary-weighted d(answer margin) / d(router bias)
+ actuator reach до top-k boundary
+ asymmetric flip/protect weighting
+ risk/consistency penalty
```

То есть нужно двигаться к однопроходному router-margin-gradient / router-constrained importance. Delta-margin causal filter можно оставить только как диагностический sanity-check для малой группы top candidates, но не как главный метод, который стоит дальше тюнить.

## 7. Router-margin-gradient importance

Конфиг:

```bash
python main.py --method v6 --mode single --domain physics --loss_top_k top_k --loss_position canonical_answer_slot --loss_type boundary_weighted_margin --margin_tau 0.0 --boundary_scale 1.0 --boundary_weight_min 0.1 --boundary_weight_max 1.0 --importance_type router_margin_grad --router_reach_temp 1.0 --router_reach_bias 0.0
```

Итоговые слои:

```text
positive: [6, 20, 22]
negative: [0, 1, 19, 21, 23]
```

Результаты:

| Эксперимент | college_physics | conceptual_physics | high_school_physics |
| ---------------------- | --------------: | -----------------: | ------------------: |
| router_margin_grad     |            0.36 |           0.506383 |            0.350993 |

Вывод: направление не подтвердилось как основной метод. Градиент по router logits дал новый сигнал, но selection снова расширился до смеси positive/negative слоев, включая поздние слои, и воспроизвел главный failure mode прошлых proxy-подходов: сильная просадка `high_school_physics`. Локальная чувствительность router margin не стала надежным предиктором generation-level accuracy.

Решение: не развивать `router_margin_grad` как основное направление. Его можно оставить как один из источников candidates, но не как финальный selector.

## 8. Generate-validate sparse selection

Конфиг:

```bash
python main.py --method v6 --mode single --domain physics --loss_top_k top_k --loss_position canonical_answer_slot --loss_type boundary_weighted_margin --margin_tau 0.0 --boundary_scale 1.0 --boundary_weight_min 0.1 --boundary_weight_max 1.0 --importance_type legacy --selection_type generate_validate --selection_validation_size 30 --selection_max_candidates 12 --selection_max_selected 3 --selection_min_correct_gain 1 --selection_negative_only
```

Candidate pool после v6 negative-only:

| Candidate          | Direction |     Delta | Validation result |
| ------------------ | --------- | --------: | ----------------- |
| Layer 19 Expert 44 | neg       | -0.158281 | 13/30, gain 0     |
| Layer 4 Expert 36  | neg       | -0.119605 | 13/30, gain 0     |
| Layer 2 Expert 27  | neg       | -0.092581 | 12/30, gain -1    |
| Layer 1 Expert 29  | neg       | -0.045194 | 11/30, gain -2    |
| Layer 5 Expert 16  | neg       | -0.030713 | 11/30, gain -2    |

Validation baseline:

```text
13/30 = 0.433333
```

Generate-validate selection:

```text
selected_count=0
positive: []
negative: []
```

Результаты:

| Эксперимент                              | college_physics | conceptual_physics | high_school_physics |
| --------------------------------------------------- | --------------: | -----------------: | ------------------: |
| generate_validate, legacy candidates, negative-only |            0.34 |           0.510638 |            0.370861 |

Сравнение с baseline:

| Dataset             |     Delta |
| ------------------- | --------: |
| college_physics     |     -0.01 |
| conceptual_physics  | +0.000638 |
| high_school_physics | -0.009139 |

Вывод: текущий запуск не дал улучшения, потому что selector не нашел ни одного candidate с положительным validation gain. Но это полезный результат для самого подхода: generate-validate правильно отверг все negative candidates, которые на реальной generation/extraction validation-метрике были нейтральны или вредны. Это ровно тот failure mode, который ломал soft causal filter и router-margin-grad: proxy-score говорил "применить", а actual generated accuracy не подтверждала пользу.

Решение: подход generate-validate стоит продолжать, но не с тем же negative-only candidate pool. Дальше нужно проверять другие источники candidates и более устойчивую validation-разметку. Selector оставляем как финальный арбитр, потому что он защищает от вредных proxy-кандидатов.

Следующие осмысленные проверки:

1. Разрешить positive+negative candidates. Negative-only pool оказался пустым по фактической пользе; возможно, полезные candidates находятся среди positive, но их нужно пропускать только через generation validation.
2. Добавить strict causal filter перед generate-validate. Лучший прошлый результат был strict causal `[2, 5]`; его стоит проверить тем же generated-accuracy арбитром.
3. Увеличить validation split до 50 или 100 samples. 30 samples могут быть слишком шумными, особенно когда ожидаемый gain равен 1-2 correct answers.
4. Использовать router-margin-grad только как candidate generator, но не как selector. Его широкий candidate pool может содержать полезные эксперты, если final decision делает generate-validate.

## 9. Generate-validate positive+negative candidates

Конфиг:

```bash
python -u main.py --method v6 --mode single --domain physics --loss_top_k top_k --loss_position canonical_answer_slot --loss_type boundary_weighted_margin --margin_tau 0.0 --boundary_scale 1.0 --boundary_weight_min 0.1 --boundary_weight_max 1.0 --importance_type legacy --selection_type generate_validate --selection_validation_size 30 --selection_max_candidates 16 --selection_max_selected 3 --selection_min_correct_gain 1
```

Candidate pool был шире, чем в negative-only запуске: 14 positive+negative candidates. Validation baseline:

```text
13/30 = 0.433333
```

Проверка candidates:

| Candidate          | Direction |     Delta | Validation result |
| ------------------ | --------- | --------: | ----------------- |
| Layer 6 Expert 23  | pos       | +0.190999 | 12/30, gain -1    |
| Layer 19 Expert 44 | neg       | -0.158281 | 12/30, gain -1    |
| Layer 6 Expert 4   | pos       | +0.156050 | 12/30, gain -1    |
| Layer 4 Expert 36  | neg       | -0.119605 | 11/30, gain -2    |
| Layer 20 Expert 36 | pos       | +0.102922 | 11/30, gain -2    |
| Layer 22 Expert 48 | pos       | +0.100825 | 11/30, gain -2    |
| Layer 2 Expert 27  | neg       | -0.092581 | 12/30, gain -1    |
| Layer 23 Expert 21 | pos       | +0.088477 | 12/30, gain -1    |
| Layer 21 Expert 23 | pos       | +0.062568 | 12/30, gain -1    |
| Layer 20 Expert 26 | pos       | +0.050947 | 12/30, gain -1    |
| Layer 1 Expert 29  | neg       | -0.045194 | 11/30, gain -2    |
| Layer 21 Expert 44 | pos       | +0.038577 | 13/30, gain 0     |
| Layer 5 Expert 16  | neg       | -0.030713 | 12/30, gain -1    |
| Layer 0 Expert 39  | pos       | +0.028602 | 12/30, gain -1    |

Generate-validate selection:

```text
selected_count=0
positive: []
negative: []
```

Финальные `AFTER` результаты не пересчитывались, потому что ни один candidate не достиг `min_correct_gain=1`, и bias не был применен.

Вывод: расширение candidate pool до positive+negative не решило проблему. Ни один эксперт с исходным delta не улучшил реальную generation/extraction validation accuracy. Это усиливает вывод, что текущий proxy selection выбирает кандидатов, которые выглядят значимыми по формуле importance, но не дают фактической пользы при применении router bias.

Следующий шаг после этого вывода: проверять не только expert, но и пару `(expert, delta_multiplier)`. Для этого в код добавлен `--selection_delta_multipliers`, чтобы понять, проблема в выборе экспертов или в масштабе/знаке bias.

## 10. Generate-validate positive+negative delta sweep, partial log

Конфиг:

```bash
python -u main.py --method v6 --mode single --domain physics --loss_top_k top_k --loss_position canonical_answer_slot --loss_type boundary_weighted_margin --margin_tau 0.0 --boundary_scale 1.0 --boundary_weight_min 0.1 --boundary_weight_max 1.0 --importance_type legacy --selection_type generate_validate --selection_validation_size 30 --selection_max_candidates 14 --selection_max_selected 3 --selection_min_correct_gain 1 --selection_delta_multipliers=-2,-1,-0.5,0.5,1,2
```

Лог `generate_validate_posneg_delta_sweep_logs.log` неполный, но уже содержит достаточно проверок для вывода:

```text
validation baseline: 13/30 = 0.433333
evaluated candidate/multiplier pairs: 129
positive gain pairs: 2
zero gain pairs: 13
negative gain pairs: 114
max gain: +1
min gain: -3
```

Лучшие найденные пары:

| Candidate          | Base direction | Base delta | Multiplier | Tested delta | Validation result |
| ------------------ | -------------- | ---------: | ---------: | -----------: | ----------------- |
| Layer 19 Expert 44 | neg            |  -0.158281 |        2.0 |    -0.316562 | 14/30, gain +1    |
| Layer 21 Expert 44 | pos            |  +0.038577 |       -0.5 |    -0.019289 | 14/30, gain +1    |

Первая пара была принята greedy selector:

```text
Accept: Layer 19 Expert 44, neg, multiplier=2.0, delta=-0.316562
validation: 13/30 -> 14/30
```

Вывод: масштаб и знак действительно имеют значение, потому что исходный delta для Layer 19 Expert 44 не давал улучшения, а `2x` дал `+1` correct на validation split. Однако это не выглядит основной причиной провалов. Если бы главная проблема была только в масштабе, sweep должен был бы находить много положительных вариантов среди тех же experts. Вместо этого из 129 проверенных пар только 2 дали `gain_correct=+1`, а 114 ухудшили результат.

Главный вывод: проблема в первую очередь в candidate generation / выборе экспертов. Текущая legacy importance на canonical answer slot выдает candidates, которые почти всегда вредны для реальной generation/extraction accuracy даже после перебора знака и масштаба. Delta sweep полезен как финальная защита и иногда может подобрать рабочий scale, но он не чинит источник плохих candidates.

Следующее направление: менять не только selection scale, а источник importance. Наиболее логичная гипотеза теперь - считать candidates на фактической generation trajectory, а не на canonical answer slot:

```text
loss_position = extracted_generated_answer
loss_type = boundary_weighted_margin или soft_margin
selection_type = generate_validate
selection_delta_multipliers = -2,-1,-0.5,0.5,1,2
```

Почему: финальная метрика зависит от autoregressive generation и extraction. Canonical answer slot может быть слишком искусственным: он оценивает "что модель сказала бы сразу после Answer:", но не то, какие experts реально управляют траекторией генерации ответа. Старый режим `extracted_generated_answer` как раз ближе к evaluation path: он генерирует ответ, находит extracted answer position и считает loss на позиции, где модель фактически начала выдавать вариант.

## 11. Next-token forced-choice generate-validate

Конфиг:

```bash
python -u main.py --method v6 --mode single --domain physics --eval_mode next_token_pred --loss_top_k top_k --loss_position canonical_answer_slot --loss_type boundary_weighted_margin --margin_tau 0.0 --boundary_scale 1.0 --boundary_weight_min 0.1 --boundary_weight_max 1.0 --importance_type legacy --selection_type generate_validate --selection_validation_size 100 --selection_max_candidates 14 --selection_max_selected 3 --selection_min_correct_gain 1 --selection_delta_multipliers=-2,-1,-0.5,0.5,1,2
```

Этот запуск проверял forced-choice метрику, согласованную с canonical answer-slot loss:

```text
eval_mode = next_token_pred
validation_size = 100
baseline validation = 33/100 = 0.330000
evaluated candidate/multiplier pairs = 162
positive gain pairs = 7
zero gain pairs = 63
negative gain pairs = 92
max gain = +2
min gain = -3
```

Лучшие одиночные пары на первом шаге:

| Candidate         | Base direction | Base delta | Multiplier | Tested delta | Validation result |
| ----------------- | -------------- | ---------: | ---------: | -----------: | ----------------- |
| Layer 6 Expert 4  | pos            |  +0.156050 |       -1.0 |    -0.156050 | 35/100, gain +2   |
| Layer 6 Expert 23 | pos            |  +0.190999 |       -2.0 |    -0.381999 | 34/100, gain +1   |
| Layer 6 Expert 23 | pos            |  +0.190999 |       +1.0 |    +0.190999 | 34/100, gain +1   |
| Layer 6 Expert 23 | pos            |  +0.190999 |       +2.0 |    +0.381999 | 34/100, gain +1   |
| Layer 4 Expert 36 | neg            |  -0.119605 |       -1.0 |    +0.119605 | 34/100, gain +1   |
| Layer 4 Expert 36 | neg            |  -0.119605 |       +0.5 |    -0.059803 | 34/100, gain +1   |
| Layer 1 Expert 29 | neg            |  -0.045194 |       +0.5 |    -0.022597 | 34/100, gain +1   |

Greedy selector принял:

```text
Layer 6 Expert 4
base direction: pos
base_delta: +0.156050
multiplier: -1.0
tested_delta: -0.156050
validation: 33/100 -> 35/100
```

После принятия этого кандидата второй шаг не нашел нового улучшения:

```text
selected_count = 1
final validation = 35/100 = 0.350000
positive layers = [6]
negative layers = []
```

Финальная forced-choice оценка:

| Dataset             | Accuracy |
| ------------------- | -------: |
| college_physics     |     0.34 |
| conceptual_physics  | 0.510638 |
| high_school_physics | 0.377483 |

Вывод: выравнивание loss/evaluation через `next_token_pred` действительно сделало сигнал чище: нашлись положительные пары, и лучший candidate дал `+2` correct на validation. Это значит, что проблема не только в том, что эксперты полностью неверные. Но ключевой факт - лучший кандидат сработал с обратным знаком относительно исходной direction: expert был выбран как positive, а полезный `tested_delta` был negative.

Главный вывод: текущая формула часто находит чувствительные/релевантные места, но неверно определяет направление и масштаб bias. При этом качество candidate generation всё еще слабое: 92 из 162 пар ухудшили результат, а после первого принятого кандидата новых улучшений не нашлось.

Отдельная техническая тревога: generate-validate внутри selection показал `35/100`, но финальная `AFTER college_physics` дала `0.34`. Это маленькое, но важное расхождение на том же forced-choice режиме. Возможная причина - in-place add/remove bias updates в fp16 во время проверки множества candidates может оставлять численный остаток или порядок вычислений отличается. Перед следующими выводами нужно сделать selection более чистым: сохранять исходный gate.bias перед каждой probe-проверкой и восстанавливать его exact copy после probe, а не полагаться на `+= delta` затем `-= delta`.

Следующий шаг:

1. Исправить probe-механику generate-validate на exact restore gate bias.
2. Повторить `next_token_pred` delta sweep.
3. Если результат сохранится, добавить direction-agnostic selection как основной режим: candidate direction становится только источником `base_delta`, а selector выбирает знак и масштаб через validation.
4. После этого проверить, переносится ли forced-choice улучшение в generation eval. Если нет, значит forced-choice и generation являются разными задачами, и нужно отдельно строить trajectory/extracted-generated candidate generation.

## 12. Next-token forced-choice generate-validate after exact bias restore

Конфиг:

```bash
python -u main.py --method v6 --mode single --domain physics --eval_mode next_token_pred --loss_top_k top_k --loss_position canonical_answer_slot --loss_type boundary_weighted_margin --margin_tau 0.0 --boundary_scale 1.0 --boundary_weight_min 0.1 --boundary_weight_max 1.0 --importance_type legacy --selection_type generate_validate --selection_validation_size 100 --selection_max_candidates 14 --selection_max_selected 3 --selection_min_correct_gain 1 --selection_delta_multipliers=-2,-1,-0.5,0.5,1,2
```

Этот запуск повторил previous `next_token_pred` delta sweep после исправления probe-механики: каждый временный bias теперь проверяется через exact snapshot/restore `gate.bias`, а не через последовательное `+= delta` / `-= delta`.

Ключевые числа из `next_token_pred_generate_validate_restore_logs.log`:

```text
LossSummary top1 correct: 33/100 = 0.330000
GenerateValidate baseline validation: 33/100 = 0.330000
candidate_count: 14
delta_multipliers: [-2, -1, -0.5, 0.5, 1, 2]
evaluated candidate/multiplier pairs: 162
positive gain pairs: 7
zero gain pairs: 63
negative gain pairs: 92
max gain: +2
min gain: -3
```

Лучший найденный кандидат:

| Candidate        | Formula direction | Base delta | Multiplier | Applied delta | Validation       |
| ---------------- | ----------------- | ---------: | ---------: | ------------: | ---------------- |
| Layer 6 Expert 4 | pos               |  +0.156050 |       -1.0 |     -0.156050 | 33/100 -> 35/100 |

После принятия этого кандидата второй greedy step не нашел нового улучшения:

```text
selected_count = 1
final_validation_correct = 35/100 = 0.350000
selected candidates = [(6, 4, 'pos', 0.15605029463768005, -1.0, -0.15605029463768005)]
```

Финальная forced-choice оценка после применения bias:

| Dataset             | Accuracy |
| ------------------- | -------: |
| college_physics     |     0.34 |
| conceptual_physics  | 0.510638 |
| high_school_physics | 0.377483 |

Сравнение с baseline:

| Metric                                                  |                                   Baseline |          Best/After |                   Delta |
| ------------------------------------------------------- | -----------------------------------------: | ------------------: | ----------------------: |
| validation next-token, college split                    |                                   0.330000 |            0.350000 |               +0.020000 |
| final AFTER college_physics next-token                  | 0.330000 inferred from validation baseline |            0.340000 |               +0.010000 |
| project baseline college_physics, generation metric     |                                   0.350000 | 0.340000 next-token | not directly comparable |
| project baseline conceptual_physics, generation metric  |                                   0.510000 | 0.510638 next-token |            roughly flat |
| project baseline high_school_physics, generation metric |                                   0.380000 | 0.377483 next-token |            roughly flat |

Уточненный окончательный вывод по этому эксперименту: первичная проблема - некорректный выбор экспертов текущей формулой. Из 14 изначально выбранных экспертов только один дал хоть сколько-то полезный результат на validation sweep; остальные не стали надежно полезными ни при одном проверенном delta. Это сильнее, чем проблема масштаба: если бы выбранные эксперты были в целом правильными, но плохо масштабированными, sweep по `[-2, -1, -0.5, 0.5, 1, 2]` должен был бы находить больше рабочих комбинаций. Вместо этого из 162 проверок только 7 дали положительный gain, 92 ухудшили результат, и после первого принятого кандидата дополнительных улучшений не нашлось.

Проблема direction/scale тоже реальна, но она вторичная по отношению к candidate generation. Лучший кандидат был найден формулой как `pos`, но полезный applied delta оказался отрицательным. Это значит, что даже когда формула случайно попадает в чувствительного эксперта, она не умеет надежно переводить importance в знак bias. Но главный провал - то, что почти весь top candidate set оказался бесполезным.

Практическая интерпретация:

1. Legacy importance в текущем виде плохо выбирает экспертов: top-14 почти целиком состоит из кандидатов, которые не улучшают validation ни при одном проверенном масштабе.
2. Legacy importance также нельзя использовать как прямой знак и величину bias. Даже единственный полезный кандидат сработал с обратным знаком.
3. Direction-agnostic validation sweep полезен как страховка, но он не решает главную проблему: если candidate generation плохой, sweep просто доказывает, что большинство кандидатов вредные.
4. Сохранилось расхождение `final_validation_correct=35/100` внутри selector и финального `AFTER college_physics=0.34`. После exact restore это уже не похоже на простую fp16 add/remove ошибку. Перед дальнейшими большими экспериментами нужно добавить явный recompute final validation сразу после завершения selection на том же validation dataset и печатать расхождение между cached/current score и recomputed score.

Главные проблемы, которые нужно решать дальше:

1. Проверочная надежность selector-а. Пока `35/100` внутри generate-validate не совпадает с final `0.34`, нельзя полностью доверять greedy decision. Это самая первая техническая проблема, потому что без нее мы не знаем, измеряем ли реальный эффект.
2. Качество candidate generation. Это главная исследовательская проблема: текущая формула дает слишком много вредных top candidates. Нужно переходить от "expert contribution to loss" к "router-action effect on answer margin", то есть оценивать не просто вклад expert output, а влияние изменения router logits/gate bias на forced-choice margin.
3. Direction/scale как отдельная задача. Формула importance не должна диктовать знак bias; знак должен выбираться validation sweep или отдельной router-action derivative формулой.
4. Transfer beyond calibration split. Даже если candidate дает +2 на college validation, перенос на conceptual/high_school почти нулевой. Значит, следующая версия должна валидироваться либо на смешанном physics validation split, либо иметь отдельный transfer check до финального вывода.

Следующий правильный шаг: сначала добавить diagnostic recompute final validation after selection, чтобы убрать техническую неопределенность в оценке. Затем основное исследовательское усилие нужно направить на новую формулу candidate generation: direct router-logit margin derivative, то есть оценку `d(correct_margin) / d(router_bias_expert)`, потому что именно router bias является фактическим управляющим действием. Direction-agnostic sweep оставляем как обязательный защитный слой, но не как главный метод решения.

## 13. Router-bias contrast: important correction about derivative type

Важное уточнение по эксперименту `router_bias_contrast_next_token_logs.log`: он уже был не loss-derivative экспериментом, а margin-derivative экспериментом.

Причина в реализации на момент запуска: в `compute_expert_importance` все режимы, кроме `legacy`, делали backward от:

```text
router_margin_objective = boundary_weight * smooth_margin
```

а не от answer loss. Поэтому фактически протестированная формула была:

```text
I_e = - <d(boundary_weight * smooth_margin)/dy, w_e * (h_e - y)>
```

а не:

```text
I_e = - <dL/dy, w_e * (h_e - y)>
```

Это важно для интерпретации: улучшение candidate quality в `router_bias_contrast_next_token_logs.log` относится именно к margin derivative / router-bias contrast подходу.

Ключевые результаты этого margin-derivative contrast запуска:

```text
baseline validation: 33/100 = 0.330000
candidate_count: 14
evaluated candidate/multiplier pairs: 162
positive gain pairs: 12
zero gain pairs: 124
negative gain pairs: 26
max gain: +1
min gain: -3
selected_count: 1
final validation: 34/100 = 0.340000
selected: Layer 3 Expert 41, multiplier=-0.5, delta=-0.097996
```

Сравнение с legacy restore:

| Metric                         | Legacy restore | Router-bias contrast margin |
| ------------------------------ | -------------: | --------------------------: |
| positive gain pairs            |          7/162 |                      12/162 |
| zero gain pairs                |         63/162 |                     124/162 |
| negative gain pairs            |         92/162 |                      26/162 |
| useful candidates              |           4/14 |                        5/14 |
| harmful/no-positive candidates |           3/14 |                        0/14 |
| best validation gain           |             +2 |                          +1 |

Итоговые next-token результаты:

| Dataset             | Legacy restore | Router-bias contrast margin |
| ------------------- | -------------: | --------------------------: |
| college_physics     |       0.340000 |                    0.340000 |
| conceptual_physics  |       0.510638 |                    0.523404 |
| high_school_physics |       0.377483 |                    0.390728 |

Вывод: margin-derivative contrast подход явно улучшил качество candidate generation: вредных probe-вариантов стало намного меньше, полезных вариантов стало больше, и полностью вредных кандидатов среди top-14 не осталось. Но максимальный validation gain стал меньше (`+1` вместо `+2`), а college accuracy не выросла выше `0.34`. Поэтому подход перспективный как направление формулы, но не финальный.

После этого в `main.py` режимы были разведены явно:

```text
router_bias_contrast_margin  -> backward от boundary_weight * smooth_margin
router_bias_contrast_loss    -> backward от answer loss
router_bias_contrast         -> alias старого margin-поведения для совместимости
```

Следующий эксперимент должен проверить именно loss derivative:

```text
I_e = - <dL/dy, w_e * (h_e - y)>
```

Это нужно, чтобы понять, что именно дало улучшение: contrast-геометрия `w_e * (h_e - y)`, margin objective, или их сочетание.

## 14. Router-bias contrast loss vs margin derivative

Сравнивались два contrast-подхода:

```text
router_bias_contrast_margin:
I_e = - <d(boundary_weight * smooth_margin)/dy, w_e * (h_e - y)>

router_bias_contrast_loss:
I_e = - <dL/dy, w_e * (h_e - y)>
```

Оба запуска имели одинаковый baseline на validation:

```text
LossSummary top1 correct: 33/100 = 0.330000
GenerateValidate baseline: 33/100 = 0.330000
```

Candidate/probe statistics:

| Metric                                   | Margin derivative | Loss derivative |
| ---------------------------------------- | ----------------: | --------------: |
| initial candidates                       |                14 |               9 |
| step-1 probe pairs                       |                84 |              54 |
| step-1 positive pairs                    |                12 |               3 |
| step-1 zero pairs                        |                69 |              50 |
| step-1 negative pairs                    |                 3 |               1 |
| useful candidates on step 1              |                 5 |               1 |
| neutral candidates on step 1             |                 9 |               7 |
| harmful/no-positive candidates on step 1 |                 0 |               1 |
| all evaluated pairs                      |               162 |             144 |
| all positive pairs                       |                12 |               5 |
| all zero pairs                           |               124 |             123 |
| all negative pairs                       |                26 |              16 |

Greedy selected candidates:

| Method            | Selected candidates                                                    | Final validation |
| ----------------- | ---------------------------------------------------------------------- | ---------------: |
| margin derivative | Layer 3 Expert 41, delta=-0.097996                                     |           34/100 |
| loss derivative   | Layer 6 Expert 23, delta=-0.381999; Layer 0 Expert 39, delta=-0.028602 |           35/100 |

Final next-token dataset results:

| Dataset              | Margin derivative | Loss derivative |
| -------------------- | ----------------: | --------------: |
| college_physics      |          0.340000 |        0.350000 |
| conceptual_physics   |          0.523404 |        0.502128 |
| high_school_physics  |          0.390728 |        0.377483 |
| mean over 3 datasets |          0.418044 |        0.409870 |

Comparison against legacy restore:

| Dataset             | Legacy restore | Margin derivative | Loss derivative |
| ------------------- | -------------: | ----------------: | --------------: |
| college_physics     |       0.340000 |          0.340000 |        0.350000 |
| conceptual_physics  |       0.510638 |          0.523404 |        0.502128 |
| high_school_physics |       0.377483 |          0.390728 |        0.377483 |

Вывод: loss derivative лучше подогнал calibration/college split: он принял два кандидата и поднял validation до `35/100`, а final `college_physics` до `0.35`. Но по качеству candidate generation он хуже margin derivative: только `1/9` initial candidates оказался полезным на первом шаге, против `5/14` у margin derivative. Кроме того, loss derivative ухудшил transfer на `conceptual_physics` и не улучшил `high_school_physics`.

Margin derivative выглядит более правильным направлением для общей формулы выбора экспертов: он дает больше полезных кандидатов, почти не генерирует вредные probe-варианты на первом шаге, и переносится лучше на другие physics datasets. Loss derivative полезен как calibration-fitting baseline, но текущие числа говорят, что он сильнее переобучается на validation/college и хуже отражает общую физическую способность модели.

Рабочий вывод: для дальнейшего развития стоит брать `router_bias_contrast_margin` как основную ветку формулы, а `router_bias_contrast_loss` держать как контрольный режим. Следующее улучшение должно быть не в возврате к loss derivative, а в улучшении validation/selection поверх margin derivative: mixed physics validation split, recompute final validation after selection, и более аккуратный выбор нескольких кандидатов без разрушения transfer.

## 15. Router-bias contrast margin direct top-5

Проверялся честный direct-режим без validation-oracle:

```text
importance_type = router_bias_contrast_margin
selection_type = direct
direct_max_candidates = 5
```

Direct применил top-5 кандидатов по `abs(delta)`:

| Rank | Candidate          | Direction |     Delta |
| ---: | ------------------ | --------- | --------: |
|    0 | Layer 3 Expert 41  | pos       | +0.195991 |
|    1 | Layer 19 Expert 44 | pos       | +0.158281 |
|    2 | Layer 7 Expert 13  | pos       | +0.155482 |
|    3 | Layer 4 Expert 36  | pos       | +0.119605 |
|    4 | Layer 19 Expert 5  | pos       | +0.102938 |

Результаты:

| Dataset             | Direct top-5 | Generate-validate top-5 | Difference |
| ------------------- | -----------: | ----------------------: | ---------: |
| college_physics     |     0.330000 |                0.340000 |  -0.010000 |
| conceptual_physics  |     0.514894 |                0.523404 |  -0.008511 |
| high_school_physics |     0.390728 |                0.390728 |   0.000000 |
| mean over 3         |     0.411874 |                0.418044 |  -0.006170 |

Сравнение с актуальным next-token baseline:

| Dataset | Next-token baseline | Direct top-5 | Delta |
| --- | ---: | ---: |
| college_physics | 0.330000 | 0.330000 | +0.000000 |
| conceptual_physics | 0.514894 | 0.514894 | +0.000000 |
| high_school_physics | 0.384106 | 0.390728 | +0.006623 |

Вывод: top-5 ranking сам по себе остается полезным, но direct-применение всех пяти raw deltas дает очень слабый эффект относительно честного next-token baseline. Из top-5 действительно было `3/5` useful на validation-probe, но direct применяет также neutral candidates и применяет base delta без выбора знака/масштаба. В итоге `college_physics` и `conceptual_physics` не меняются относительно baseline, а прирост есть только на `high_school_physics`.

Главная интерпретация: текущая формула уже неплохо ранжирует экспертов, но action policy все еще слабая. Проблема теперь не столько "найти кандидатов", сколько "сколько кандидатов применять и с каким знаком/масштабом без validation". Generate-validate top-5 выбрал только один кандидат, а direct top-5 применил сразу пять; именно это, вероятно, внесло лишний шум и конфликтующие bias updates.

Следующий честный direct-тест без validation-oracle должен быть `direct_max_candidates=1`, затем `direct_max_candidates=3`. Если top-1 сохранит conceptual/high_school и не уронит college, значит финальное правило должно быть агрессивно sparse: брать только самый сильный candidate, а не весь top-5.

## 16. Correct physics next-token baseline

После исправления baseline-режима в `main.py` был перезапущен чистый baseline:

```bash
python -u main.py --method baseline --mode single --domain physics --eval_mode next_token_pred
```

Актуальный baseline для всех последующих `next_token_pred` сравнений:

| Dataset             | Baseline |
| ------------------- | -------: |
| college_physics     | 0.330000 |
| conceptual_physics  | 0.514894 |
| high_school_physics | 0.384106 |

Важно: прежний project baseline `0.35 / 0.51 / 0.38` был generation/evaluation ориентиром из `project_description.md`, а не чистым `next_token_pred` baseline. Для текущей серии forced-choice экспериментов корректная опора - таблица выше.

Пересчет ключевых последних результатов относительно нового baseline:

| Experiment                                          | college_physics | conceptual_physics | high_school_physics | Mean delta |
| --------------------------------------------------- | --------------: | -----------------: | ------------------: | ---------: |
| router_bias_contrast_margin direct top-5            |       +0.000000 |          +0.000000 |           +0.006623 |  +0.002208 |
| router_bias_contrast_margin generate-validate top-5 |       +0.010000 |          +0.008511 |           +0.006623 |  +0.008378 |
| router_bias_contrast_loss generate-validate         |       +0.020000 |          -0.012766 |           -0.006623 |  +0.000204 |

Вывод после корректировки baseline: direct top-5 не является провалом, но эффект слишком слабый - он фактически сохраняет baseline на двух датасетах и немного улучшает `high_school_physics`. Generate-validate top-5 показывает, что в top-5 есть полезный сигнал, но финальный direct policy пока не умеет извлекать его без validation-oracle. Главная следующая задача остается прежней: сделать не-валидационное правило выбора знака/масштаба/числа применяемых кандидатов.

## 17. Router-bias contrast margin direct top-3 with negative half scale

Проверен честный direct-режим без validation-oracle:

```bash
python -u main.py --method v6 --mode single --domain physics --eval_mode next_token_pred --loss_top_k top_k --loss_position canonical_answer_slot --loss_type boundary_weighted_margin --margin_tau 0.0 --boundary_scale 1.0 --boundary_weight_min 0.1 --boundary_weight_max 1.0 --importance_type router_bias_contrast_margin --selection_type direct --direct_max_candidates 3 --direct_delta_multiplier -0.5
```

Direct top-3 применил только три strongest candidates по `abs(delta)`, но с обратным половинным масштабом:

| Rank | Candidate          | Base delta | Applied delta |
| ---: | ------------------ | ---------: | ------------: |
|    0 | Layer 3 Expert 41  |  +0.195991 |     -0.097996 |
|    1 | Layer 19 Expert 44 |  +0.158281 |     -0.079141 |
|    2 | Layer 7 Expert 13  |  +0.155482 |     -0.077741 |

Результаты относительно актуального `next_token_pred` baseline:

| Dataset             | Baseline | Direct top-3, multiplier=-0.5 |     Delta |
| ------------------- | -------: | ----------------------------: | --------: |
| college_physics     | 0.330000 |                      0.330000 | +0.000000 |
| conceptual_physics  | 0.514894 |                      0.519149 | +0.004255 |
| high_school_physics | 0.384106 |                      0.390728 | +0.006623 |
| mean over 3         | 0.409667 |                      0.413292 | +0.003626 |

Сравнение с direct top-5 raw:

| Dataset             | Direct top-5 raw | Direct top-3, multiplier=-0.5 |     Delta |
| ------------------- | ---------------: | ----------------------------: | --------: |
| college_physics     |         0.330000 |                      0.330000 | +0.000000 |
| conceptual_physics  |         0.514894 |                      0.519149 | +0.004255 |
| high_school_physics |         0.390728 |                      0.390728 | +0.000000 |
| mean over 3         |         0.411874 |                      0.413292 | +0.001418 |

Вывод: эксперимент сработал умеренно положительно. Он не достиг уровня generate-validate top-5 (`mean=0.418044`), но улучшил честный direct top-5 raw и дал прирост к baseline на двух transfer datasets без падения `college_physics`. Это усиливает гипотезу, что `router_bias_contrast_margin` уже находит достаточно осмысленных кандидатов, а слабое место сейчас - action policy: знак, масштаб и число применяемых bias updates.

Отдельно важен знак. Для margin-цели математически естественное действие - gradient ascent по margin, тогда как текущая историческая формула использует знак, унаследованный от loss-minimization. То, что `direct_delta_multiplier=-0.5` улучшает результат, является практическим сигналом, что direct-применение текущих positive deltas с `+1.0` имеет неверный или слишком агрессивный знак/масштаб. Следующий шаг должен быть не ручной перебор `-0.5`, а вычисляемая политика `delta`, например sign-corrected margin-ascent или damped/Fisher-scaled update.

## 18. Router-bias contrast margin direct top-3 with fixed Fisher-style delta

Проверен вычисляемый direct policy:

```bash
python -u main.py --method v6 --mode single --domain physics --eval_mode next_token_pred --loss_top_k top_k --loss_position canonical_answer_slot --loss_type boundary_weighted_margin --margin_tau 0.0 --boundary_scale 1.0 --boundary_weight_min 0.1 --boundary_weight_max 1.0 --importance_type router_bias_contrast_margin --selection_type direct --direct_max_candidates 3 --direct_delta_policy fisher_margin --direct_delta_eta 1.0 --direct_delta_l2 0.001 --direct_delta_gap_cap 0.25
```

Applied deltas:

| Rank | Candidate          | Base delta | Fisher raw step |      Cap | Applied delta |
| ---: | ------------------ | ---------: | --------------: | -------: | ------------: |
|    0 | Layer 3 Expert 41  |  +0.195991 |       -2.369422 | 0.122129 |     -0.122129 |
|    1 | Layer 19 Expert 44 |  +0.158281 |       -2.200544 | 0.056914 |     -0.056914 |
|    2 | Layer 7 Expert 13  |  +0.155482 |       -1.356846 | 0.082944 |     -0.082944 |

Результаты:

| Dataset             | Baseline | Fisher fixed |     Delta |
| ------------------- | -------: | -----------: | --------: |
| college_physics     | 0.330000 |     0.330000 | +0.000000 |
| conceptual_physics  | 0.514894 |     0.514894 | +0.000000 |
| high_school_physics | 0.384106 |     0.384106 | +0.000000 |
| mean over 3         | 0.409667 |     0.409667 | +0.000000 |

Сравнение с лучшим честным direct top-3:

| Dataset             | Direct top-3, multiplier=-0.5 | Fisher fixed |
| ------------------- | ----------------------------: | -----------: |
| college_physics     |                      0.330000 |     0.330000 |
| conceptual_physics  |                      0.519149 |     0.514894 |
| high_school_physics |                      0.390728 |     0.384106 |
| mean over 3         |                      0.413292 |     0.409667 |

Вывод: текущий Fisher-style delta не сработал. Все `raw_step` оказались намного больше router-gap cap, поэтому фактически policy превратился не в curvature-aware update, а в `sign-corrected gap-cap update`. Из-за этого он потерял полезную относительную калибровку base delta: Layer 3 был усилен сильнее успешного `-0.5` варианта, Layer 19 был ослаблен слабее успешного варианта, Layer 7 оказался близко. Такая переразметка масштабов убрала весь положительный transfer effect.

Практический вывод: направление "вычислять delta" остается нужным, но именно формула `sum / sum_sq` в текущем виде не годится как final action policy. Более перспективно использовать base delta как уже откалиброванную router-gap величину и вычислять только знак/глобальное damping-правило, либо строить per-candidate scale из устойчивости/консистентности сигнала без насыщения в gap cap.

## 19. Router-bias contrast margin direct top-3 with stability-damped delta

Проверен stability-based direct policy:

```bash
python -u main.py --method v6 --mode single --domain physics --eval_mode next_token_pred --loss_top_k top_k --loss_position canonical_answer_slot --loss_type boundary_weighted_margin --margin_tau 0.0 --boundary_scale 1.0 --boundary_weight_min 0.1 --boundary_weight_max 1.0 --importance_type router_bias_contrast_margin --selection_type direct --direct_max_candidates 3 --direct_delta_policy stability_damped --direct_delta_min_stability 0.0 --direct_delta_strength_temp 1.0 --direct_delta_max_damping 0.5
```

Applied deltas:

| Rank | Candidate          | Base delta | Cancellation | Strength norm |  Damping | Applied delta |
| ---: | ------------------ | ---------: | -----------: | ------------: | -------: | ------------: |
|    0 | Layer 3 Expert 41  |  +0.195991 |     0.122473 |      0.496487 | 0.030403 |     -0.005959 |
|    1 | Layer 19 Expert 44 |  +0.158281 |     0.100301 |      0.489636 | 0.024556 |     -0.003887 |
|    2 | Layer 7 Expert 13  |  +0.155482 |     0.478156 |      1.000000 | 0.239078 |     -0.037172 |

Результаты:

| Dataset             | Baseline | Stability-damped |     Delta |
| ------------------- | -------: | ---------------: | --------: |
| college_physics     | 0.330000 |         0.330000 | +0.000000 |
| conceptual_physics  | 0.514894 |         0.514894 | +0.000000 |
| high_school_physics | 0.384106 |         0.390728 | +0.006623 |
| mean over 3         | 0.409667 |         0.411874 | +0.002208 |

Сравнение с лучшим direct top-3:

| Dataset             | Direct top-3, multiplier=-0.5 | Stability-damped |
| ------------------- | ----------------------------: | ---------------: |
| college_physics     |                      0.330000 |         0.330000 |
| conceptual_physics  |                      0.519149 |         0.514894 |
| high_school_physics |                      0.390728 |         0.390728 |
| mean over 3         |                      0.413292 |         0.411874 |

Вывод: stability-damped не стал лучше текущего лучшего direct top-3 `-0.5`. Он лучше Fisher fixed, потому что сохранил прирост на `high_school_physics`, но хуже ручного `-0.5`, потому что потерял прирост на `conceptual_physics`.

Главная диагностика: sign-cancellation оказался слишком жестким критерием. Он почти занулил Layer 3 Expert 41 и Layer 19 Expert 44 (`damping=0.0304` и `0.0246`), хотя Layer 3 Expert 41 ранее был лучшим generate-validate кандидатом и давал полезный эффект при delta около `-0.098`. Значит per-sample знаковая стабильность градиентного сигнала плохо коррелирует с полезностью глобального router-bias update: полезный доменный эксперт может иметь mixed per-sample contributions, но все равно улучшать итоговую forced-choice accuracy.

Практический вывод: дальше не стоит усиливать sign-cancellation как основной фильтр. Более правильный следующий шаг - оставить глобальный sign-corrected base delta и искать не per-candidate damping через cancellation, а один вычисляемый глобальный damping/temperature для top-N, либо перейти к top-1/top-2 sparse policy. Текущий лучший честный direct остается `router_bias_contrast_margin + direct_max_candidates=3 + direct_delta_multiplier=-0.5`.

## 20. Router-bias contrast margin direct top-3 with gap-safe global alpha

Проверен global alpha policy:

```bash
python -u main.py --method v6 --mode single --domain physics --eval_mode next_token_pred --loss_top_k top_k --loss_position canonical_answer_slot --loss_type boundary_weighted_margin --margin_tau 0.0 --boundary_scale 1.0 --boundary_weight_min 0.1 --boundary_weight_max 1.0 --importance_type router_bias_contrast_margin --selection_type direct --direct_max_candidates 3 --direct_delta_policy gap_safe_global --direct_delta_gap_fraction 0.4 --direct_delta_alpha_max 0.5
```

Policy вычислила:

```text
selected_min_safe_gap = 0.227656
selected_max_abs_delta = 0.195991
gap_fraction = 0.4
raw_alpha = alpha = 0.464626
```

Applied deltas:

| Rank | Candidate          | Base delta | Applied delta |
| ---: | ------------------ | ---------: | ------------: |
|    0 | Layer 3 Expert 41  |  +0.195991 |     -0.091063 |
|    1 | Layer 19 Expert 44 |  +0.158281 |     -0.073542 |
|    2 | Layer 7 Expert 13  |  +0.155482 |     -0.072241 |

Результаты:

| Dataset             | Baseline | Gap-safe global |     Delta |
| ------------------- | -------: | --------------: | --------: |
| college_physics     | 0.330000 |        0.330000 | +0.000000 |
| conceptual_physics  | 0.514894 |        0.514894 | +0.000000 |
| high_school_physics | 0.384106 |        0.384106 | +0.000000 |
| mean over 3         | 0.409667 |        0.409667 | +0.000000 |

Сравнение с лучшим direct top-3:

| Dataset             | Direct top-3, multiplier=-0.5 | Gap-safe global |
| ------------------- | ----------------------------: | --------------: |
| college_physics     |                      0.330000 |        0.330000 |
| conceptual_physics  |                      0.519149 |        0.514894 |
| high_school_physics |                      0.390728 |        0.384106 |
| mean over 3         |                      0.413292 |        0.409667 |

Вывод: `gap_safe_global` не сработал. Он сохранил относительные base-delta пропорции и выбрал alpha близко к успешному ручному `0.5`, но `alpha=0.464626` оказалось недостаточно: эффект полностью откатился к baseline. Это показывает, что полезный эффект router-bias update очень дискретный и чувствительный к масштабу; небольшое уменьшение относительно `-0.5` уже может потерять boundary flips, которые давали прирост на `conceptual_physics` и `high_school_physics`.

Практический вывод: текущая серия computed-delta policy не дала лучшего результата. Fisher ломал относительные масштабы, stability занулял полезные mixed experts, gap-safe alpha оказался слишком слабым. Лучший честный direct результат по-прежнему остается ручной `direct_delta_multiplier=-0.5`. Следующая исследовательская проблема уже не просто "как вычислить alpha", а "какие конкретные примеры/границы переворачивает успешный -0.5 и почему близкие масштабы их не переворачивают". Нужно анализировать per-example answer-margin deltas для best policy против failed policies.

## 21. Router-bias contrast margin direct top-3 with full sign-corrected v6 step

Проверен полный sign-corrected v6 step:

```bash
python -u main.py --method v6 --mode single --domain physics --eval_mode next_token_pred --loss_top_k top_k --loss_position canonical_answer_slot --loss_type boundary_weighted_margin --margin_tau 0.0 --boundary_scale 1.0 --boundary_weight_min 0.1 --boundary_weight_max 1.0 --importance_type router_bias_contrast_margin --selection_type direct --direct_max_candidates 3 --direct_delta_policy multiplier --direct_delta_multiplier -1.0
```

Applied deltas:

| Rank | Candidate          | Base delta | Applied delta |
| ---: | ------------------ | ---------: | ------------: |
|    0 | Layer 3 Expert 41  |  +0.195991 |     -0.195991 |
|    1 | Layer 19 Expert 44 |  +0.158281 |     -0.158281 |
|    2 | Layer 7 Expert 13  |  +0.155482 |     -0.155482 |

Результаты:

| Dataset             | Baseline | Direct top-3, multiplier=-1.0 |     Delta |
| ------------------- | -------: | ----------------------------: | --------: |
| college_physics     | 0.330000 |                      0.330000 | +0.000000 |
| conceptual_physics  | 0.514894 |                      0.519149 | +0.004255 |
| high_school_physics | 0.384106 |                      0.370861 | -0.013245 |
| mean over 3         | 0.409667 |                      0.406670 | -0.002997 |

Сравнение с лучшим direct top-3:

| Dataset             | Direct top-3, multiplier=-0.5 | Direct top-3, multiplier=-1.0 |
| ------------------- | ----------------------------: | ----------------------------: |
| college_physics     |                      0.330000 |                      0.330000 |
| conceptual_physics  |                      0.519149 |                      0.519149 |
| high_school_physics |                      0.390728 |                      0.370861 |
| mean over 3         |                      0.413292 |                      0.406670 |

Вывод: полный sign-corrected v6 step слишком агрессивен. Он сохраняет полезный эффект на `conceptual_physics`, но ломает transfer на `high_school_physics` и уводит среднее ниже baseline. Значит `-0.5` не был просто "недодавленным" шагом; он ближе к безопасной верхней зоне масштаба. Увеличивать масштаб до полного `-1.0` нельзя.

Практический вывод по scale: масштаб действительно нужен. Лучший найденный scale сейчас не полный v6 step, а damped sign-corrected step около `0.5`. Слишком маленький scale (`0.4646` в gap-safe) не переворачивает нужные boundary cases, а слишком большой scale (`1.0`) начинает ломать другие датасеты. Это подтверждает, что основная проблема теперь - найти устойчивый способ выбирать scale в узкой рабочей зоне, а не просто "включать экспертов сильнее".
