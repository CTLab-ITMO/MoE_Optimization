# Точки расширения `common/`

Этот модуль спроектирован так, чтобы новые стратегии добавлялись **аддитивно** —
новым подклассом и записью в реестре, без форка ядра. Ниже — карта расширяемых
осей: абстрактный интерфейс, его реестр, entry-point-группа для out-of-tree
плагинов и пример регистрации.

Общая механика (см. [`registry.py`](registry.py)): на каждой оси есть generic
`Registry` (имя → инстанс), пиннутый встроенными реализациями в фиксированном
порядке. Стадии читают **только** из реестра. Сторонний пакет регистрирует плагин
фактом установки через entry point; пайплайн подхватывает его, если в конфиге
выставлен `LOAD_PLUGINS = True` (см. [`plugins.py`](plugins.py)).

`**kwargs` живёт **только на абстрактном интерфейсе** (forward-совместимость);
конкретный подкласс принимает явные именованные параметры (так ловятся опечатки).
Динамическое общее состояние (модель, слой, device, seed, данные) передаётся
единым `StageContext` ([`context.py`](context.py)), а не голыми `**kwargs`.

## Оси с реестром

| Ось | Интерфейс (ABC) | Реестр | Entry-point группа | Встроенные реализации |
|---|---|---|---|---|
| Слияние | `FusionMethod` ([fusion.py](fusion.py)) | `FUSION_REGISTRY` | `moe.fusion` | ZipIt, Average, Pruning, Procrustes, Fisher, TIES, Distillation + обёртки `LS_<name>` |
| Группировка | `Grouping` = `AffinityProvider` × `Partitioner` ([grouping.py](grouping.py)) | `GROUPING_REGISTRY` | `moe.grouping` | activation_correlation, random, output_similarity |
| Метрика | `Metric` ([metrics.py](metrics.py)) | `METRIC_REGISTRY` | `moe.metric` | MSE, RMSE, MAE, RelMSE, Cosine, Pearson, R2, CKA, JS, KL, Top1, Top5, EntropyGap |
| Схема датасета | `DatasetSchema` ([dataset_schema.py](dataset_schema.py)) | `SCHEMA_REGISTRY` | `moe.dataset_schema` | mmlu, winogrande, ARC, openbookqa |
| Активация | (фабрики `nn.Module`) | `ACTIVATION_REGISTRY` ([mlp.py](mlp.py)) | — | SiLU (дефолт), ReLU, GELU |

### Зарегистрировать свою реализацию

В дереве — декоратором/функцией:

```python
from common.fusion import FUSION_REGISTRY, FusionMethod

class MyFusion(FusionMethod):
    name = "MyFusion"
    needs_data = False
    def merge(self, mlp_a, mlp_b, train_data, create_new=True, ctx=None):
        ...  # вернуть слитый AdvancedMLP

FUSION_REGISTRY.register("MyFusion", MyFusion())
```

Вне дерева (сторонний пакет) — entry point в `pyproject.toml`, затем `LOAD_PLUGINS = True`:

```toml
[project.entry-points."moe.fusion"]
MyFusion = "my_pkg.fusion:MyFusion"
```

Коллизия имени с уже зарегистрированным — ошибка (`override=False`). Обёртки-
трансформеры (как `LS_<name>`) применяются ко **всем** зарегистрированным методам,
включая плагины (см. `apply_meta_wrappers` в [fusion.py](fusion.py)).

## Оси через DI в конфиге (модель-зависимые)

Эти выбираются **инстансом класса** в `experts_fusion/config.py` (а не строкой);
дефолты `Qwen15*` воспроизводят текущее поведение для `Qwen1.5-MoE`.

| Что | Интерфейс (ABC) | Дефолт | Поле конфига |
|---|---|---|---|
| Канонизация роутера | `RouterAdapter` ([router.py](router.py)) | `Qwen15RouterAdapter` | `ROUTER_ADAPTER` |
| Раскладка эксперта | `ExpertAdapter` ([expert_adapter.py](expert_adapter.py)) | `Qwen15ExpertAdapter` | `EXPERT_ADAPTER` |
| Shared-эксперт | `SharedExpertAdapter` ([expert_adapter.py](expert_adapter.py)) | `Qwen15SharedExpert` | — |
| Формат QA-промпта | `QAParser` ([qa_parser.py](qa_parser.py)) | `Qwen15QAParser` | `QA_PARSER` |

Чтобы поддержать другое семейство моделей — новый подкласс каждого адаптера и
подстановка инстансов в конфиге; ядро менять не нужно.

## Рантайм-контракт реконструкции

`FusedGroup.reconstruct(x, member) -> Tensor` ([fused_group.py](fused_group.py)) —
от него зависит `FusionLayer`, а не от конкретной реализации unmerge. `MLPFusedGroup`
(поверх `AdvancedMLP.forward_unzip`) — одна из реализаций; `FusedExpert` сериализует
именно этот контракт, так что новый способ реконструкции добавляется новой
реализацией без правки слоя.
