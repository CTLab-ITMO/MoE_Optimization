"""Opt-in discovery of out-of-tree plugins into the canonical registries.

A third-party package can advertise a fusion / grouping / metric / dataset-schema
implementation via an entry point (groups ``moe.fusion`` / ``moe.grouping`` /
``moe.metric`` / ``moe.dataset_schema``). Discovery is OFF by default: the in-tree
registries are pinned and a golden snapshot matches them exactly, so auto-loading
would be a loud (and correct) drift. A run opts in via config; nothing happens
unless plugins are actually installed.

``load_plugins`` is idempotent within a process -- the stages run in one process,
so calling it at the top of several stages loads the plugins once.

Library code only -- this module never imports an experiment ``config``.
"""
from common import dataset_schema, fusion, grouping, metrics

_loaded = False


def load_plugins(enabled: bool = False) -> dict:
    """Load installed plugins into each registry; no-op unless ``enabled``.

    Returns ``{group: [loaded names]}``. After loading fusion plugins it re-applies
    the meta-wrappers so a plugin's base method also gets its ``LS_<name>`` variant.
    """
    global _loaded
    if not enabled or _loaded:
        return {}
    _loaded = True
    loaded = {
        fusion.FUSION_ENTRY_POINT_GROUP: fusion.FUSION_REGISTRY.load_entry_points(
            fusion.FUSION_ENTRY_POINT_GROUP
        ),
        grouping.GROUPING_ENTRY_POINT_GROUP: grouping.GROUPING_REGISTRY.load_entry_points(
            grouping.GROUPING_ENTRY_POINT_GROUP
        ),
        metrics.METRIC_ENTRY_POINT_GROUP: metrics.METRIC_REGISTRY.load_entry_points(
            metrics.METRIC_ENTRY_POINT_GROUP
        ),
        dataset_schema.SCHEMA_ENTRY_POINT_GROUP: dataset_schema.SCHEMA_REGISTRY.load_entry_points(
            dataset_schema.SCHEMA_ENTRY_POINT_GROUP
        ),
    }
    fusion.apply_meta_wrappers(fusion.FUSION_REGISTRY)
    return loaded
