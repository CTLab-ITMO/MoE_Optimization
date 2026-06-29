"""Tests for opt-in plugin discovery (common/plugins.py).

No third-party ``moe.*`` plugin is installed in the test env, so enabling
discovery is a structural no-op: the registries stay pinned. These tests pin
the off-by-default behaviour, the group coverage, and the within-process
idempotency guard.
"""
from common import fusion, plugins


def test_load_plugins_disabled_is_noop(monkeypatch):
    monkeypatch.setattr(plugins, "_loaded", False)
    before = fusion.FUSION_REGISTRY.names()
    assert plugins.load_plugins(False) == {}
    assert fusion.FUSION_REGISTRY.names() == before


def test_load_plugins_enabled_covers_all_groups_and_keeps_registry_pinned(monkeypatch):
    monkeypatch.setattr(plugins, "_loaded", False)
    before = fusion.FUSION_REGISTRY.names()
    loaded = plugins.load_plugins(True)
    assert set(loaded) == {
        "moe.fusion",
        "moe.grouping",
        "moe.metric",
        "moe.dataset_schema",
    }
    # nothing installed -> no names loaded, the pinned in-tree registry is unchanged
    assert all(names == [] for names in loaded.values())
    assert fusion.FUSION_REGISTRY.names() == before


def test_load_plugins_is_idempotent_within_process(monkeypatch):
    monkeypatch.setattr(plugins, "_loaded", False)
    plugins.load_plugins(True)
    assert plugins.load_plugins(True) == {}  # second call is guarded
