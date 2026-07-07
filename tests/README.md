# Тесты

Unit- и интеграционные тесты для [`common/`](../common) и пакета-пайплайна
[`experts_fusion/`](../experts_fusion). Старый модуль
[`Qwen_MoE_experts_fusion_experiment/`](../Qwen_MoE_experts_fusion_experiment)
**снят с CI** (решение №6): он остаётся на диске, но в прогоне CI не участвует —
его собственные тесты (`tests/experiment/`) исключены из сбора, а тесты
эквивалентности, гоняющие legacy-код (golden `test_golden_legacy_*`,
seam-тесты `*_matches_legacy*`), помечены `skip`. Гарантией нового модуля
остаётся его golden-оракул (`test_golden_new_module` против `golden_v1.json`).

Все тесты выполняются на CPU и не требуют скачивания модели Qwen или GPU
(модель подменяется на маленький детерминированный `FakeMoE`), поэтому запускаются
локально за секунды.

## Установка зависимостей

```bash
pip install -r tests/requirements.txt
```

## Запуск

Из корня репозитория:

```bash
pytest tests
```

Конфигурация (`pytest.ini`) включает покрытие по `common` и `experts_fusion`
(legacy-модуль снят с покрытия и из сбора, решение №6); HTML-отчёт пишется в
`htmlcov/`.

## Структура

- `tests/common/` — модули `common/` (слияние, метрики, группировка, сериализация,
  адаптеры, реестры, провенанс, …).
- `tests/experts_fusion/` — конфиг, швы стадий и сквозной прогон 4-стадийного пайплайна.
- `tests/golden/` — checked-in golden (`golden_v1.json` + `harness.py`): один
  драйвер прогоняет value-стадии на `FakeMoE` и сверяет с одним снапшотом (точный
  `Partition`, реконструкционные метрики, `accuracy_list`, снапшот реестров). В CI
  под снапшот держится **новый** модуль (`test_golden_new_module` +
  `test_golden_self_consistency`). Тесты эквивалентности старому модулю
  (`test_golden_legacy_*`) сохранены, но помечены `skip` — legacy снят с CI
  (решение №6); драйвер `harness.py` по-прежнему умеет гонять оба, снять skip
  можно локально. Обновление golden — только под флагом `UPDATE_GOLDEN` с ревью
  диффа.
- `tests/experiment/` — тесты legacy-модуля `Qwen_MoE_experts_fusion_experiment`;
  исключены из сбора (`--ignore` в `pytest.ini`) после снятия legacy с CI, но
  оставлены на диске и запускаются вручную (`pytest tests/experiment`).
