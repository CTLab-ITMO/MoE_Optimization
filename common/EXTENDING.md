# Точки расширения `common/`

Модуль спроектирован так, чтобы новые стратегии добавлялись **аддитивно правкой
исходников**: пишешь новый подкласс рядом со встроенными и регистрируешь его в
реестре — без переписывания `if`-цепочек по стадиям. Ниже — карта расширяемых осей:
абстрактный интерфейс, его реестр и пример регистрации.

Общая механика (см. [`registry.py`](registry.py)): на каждой оси есть generic
`Registry` (имя → инстанс), заполняемый встроенными реализациями в фиксированном
порядке. Стадии читают **только** из реестра. Чтобы добавить своё — пишешь подкласс
и вызываешь `register(name, instance)` рядом со встроенными; коллизия имени —
явная ошибка (`override=False`).

Статическая конфигурация стратегии — её явные именованные параметры `__init__`
(дефолты сохраняют поведение). Динамическое общее состояние (модель, слой, device,
seed, данные) передаётся единым `StageContext` ([`context.py`](context.py)).

## Оси с реестром

| Ось | Интерфейс (ABC) | Реестр | Встроенные реализации |
|---|---|---|---|
| Слияние | `FusionMethod` ([fusion.py](fusion.py)) | `FUSION_REGISTRY` | ZipIt, Average, Pruning, Procrustes, Fisher, TIES, Distillation + обёртки `LS_<name>` |
| Группировка | `Grouping` = `AffinityProvider` × `Partitioner` ([grouping.py](grouping.py)) | `GROUPING_REGISTRY` | activation_correlation, random, output_similarity |
| Метрика | `Metric` ([metrics.py](metrics.py)) | `METRIC_REGISTRY` | MSE, RMSE, MAE, RelMSE, Cosine, Pearson, R2, CKA, JS, KL, Top1, Top5, EntropyGap |
| Схема датасета | `DatasetSchema` ([dataset_schema.py](dataset_schema.py)) | `SCHEMA_REGISTRY` | mmlu, winogrande, ARC, openbookqa |
| Активация | (фабрики `nn.Module`) | `ACTIVATION_REGISTRY` ([mlp.py](mlp.py)) | SiLU (дефолт), ReLU, GELU |

### Зарегистрировать свою реализацию

Пишешь подкласс и регистрируешь его рядом со встроенными:

```python
from common.fusion import FUSION_REGISTRY, FusionMethod

class MyFusion(FusionMethod):
    name = "MyFusion"
    needs_data = False
    def merge(self, mlp_a, mlp_b, train_data, create_new=True, ctx=None):
        ...  # вернуть слитый AdvancedMLP

FUSION_REGISTRY.register("MyFusion", MyFusion())
```

Арность — класс-атрибут `supported_arity`: `2` (дефолт) — строго парный метод,
реализующий `merge`; `None` — k-агностичный метод (любой k ≥ 1), переопределяющий
`merge_group(mlps, train_data, create_new=True, ctx=None)` (так сделаны `Average`
и `Pruning`).

Обёртки-трансформеры (как `LS_<name>`) применяются ко **всем** зарегистрированным
методам (см. `apply_meta_wrappers` в [fusion.py](fusion.py)) — новый базовый метод
автоматически получает свой `LS_<name>`. Отказаться от обёртки (например, для
k-агностичного метода, которому парный LS-рефит не имеет смысла) — класс-атрибут
`wrappable = False`.

> Расширение «interface может потребовать правок»: при изменении абстрактной
> сигнатуры обнови все её подклассы в дереве — здесь нет forward-совместимости для
> внешних пакетов, поэтому интерфейс держится строгим и явным.

## Оси через DI в конфиге (модель-зависимые)

Эти выбираются **инстансом класса** в `experts_fusion/config.py` (а не строкой);
дефолты `Qwen15*` воспроизводят текущее поведение для `Qwen1.5-MoE`.

| Что | Интерфейс (ABC) | Дефолт | Поле конфига |
|---|---|---|---|
| Канонизация роутера | `RouterAdapter` ([router.py](router.py)) | `Qwen15RouterAdapter` | `ROUTER_ADAPTER` |
| Раскладка эксперта | `ExpertAdapter` ([expert_adapter.py](expert_adapter.py)) | `Qwen15ExpertAdapter` | `EXPERT_ADAPTER` |
| Shared-эксперт | `SharedExpertAdapter` ([expert_adapter.py](expert_adapter.py)) | `Qwen15SharedExpertAdapter` | `SHARED_EXPERT_ADAPTER` |
| Формат QA-промпта | `QAParser` ([qa_parser.py](qa_parser.py)) | `Qwen15QAParser` | `QA_PARSER` |

Чтобы поддержать другое семейство моделей — новый подкласс каждого адаптера и
подстановка инстансов в конфиге; ядро менять не нужно. Референсное второе
семейство уже в дереве — **Mixtral**: `MixtralExpertAdapter` (ModuleList с
именами `w1`/`w3`/`w2`), `MixtralRouterAdapter` (`norm_topk_prob=True`),
`NoSharedExpertAdapter` (у Mixtral нет shared-эксперта) и `MixtralQAParser`
(один `user`-тёрн формата Mistral, `leading_space=True`). Захват роутера в
стадии 1 идёт через `RouterAdapter.route_module(mlp, x, top_k=...)` — семейство,
чей роутер зовётся не как `mlp.gate(x)`, переопределяет только этот метод.

## Рантайм-контракт реконструкции

`FusedGroup.reconstruct(x, member) -> Tensor` ([fused_group.py](fused_group.py)) —
от него зависит `FusionLayer`, а не от конкретной реализации unmerge. Реализации
собраны в `RECONSTRUCTION_REGISTRY` (тег → `ReconstructionSpec`: допустимый
`group_size` + сборка runtime-группы): `zip_unmerge` — парный `MLPFusedGroup`
(поверх `AdvancedMLP.forward_unzip`), `identity` — k-агностичный
`IdentityFusedGroup` (fused-выход как есть для любого member; k ≥ 1).
`FusedExpert` сериализует именно этот контракт (тег + `group_size` в заголовке),
так что новый способ реконструкции добавляется регистрацией новой спеки —
без правки слоя и артефакта.
