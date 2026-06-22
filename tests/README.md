# Тесты

Unit-тесты для модулей [`common/`](../common) и
[`Qwen_MoE_experts_fusion_experiment/`](../Qwen_MoE_experts_fusion_experiment).

Все тесты выполняются на CPU и не требуют скачивания модели Qwen или
наличия GPU, поэтому часть кода они не покрывают, однако позволяют запуск локально.

## Установка зависимостей

```bash
pip install -r requirements.txt
```

## Запуск

Из корня репозитория:

```bash
pytest tests
```

Для указания процента покрытия:

```bash
pytest --cov=common --cov=Qwen_MoE_experts_fusion_experiment --cov-report=html tests
```

## Покрытие

Текущее общее покрытие --- **94%** (`common/` --- 96-100% по каждому файлу).

| Файл | Покрытие |
|------|----------|
| common/mlp.py | 100% |
| common/qa_eval.py | 100% |
| common/seeding.py | 100% |
| common/fusion.py | 99% |
| common/metrics.py | 99% |
| common/pairing.py | 98% |
| common/fusion_layer.py | 98% |
| common/mlp_utils.py | 96% |
| Qwen_.../1_generate_activation_matrix.py | 97% |
| Qwen_.../2_find_best_matches.py | 100% |
| Qwen_.../3_save_experts.py | 79% |
| Qwen_.../4_write_mlp_inputs.py | 100% |
| Qwen_.../5_activations.py | 98% |
| Qwen_.../6_fusion.py | 92% |
| Qwen_.../7_final_benchmark.py | 46% |
| Qwen_.../8_generation.py | 67% |
| Qwen_.../9_test.py | 94% |
| Qwen_.../config.py | 100% |
