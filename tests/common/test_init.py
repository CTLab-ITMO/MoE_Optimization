"""The common package re-exports its extension points (see common/__init__.py)."""
import common


def test_all_exports_are_importable():
    for name in common.__all__:
        assert hasattr(common, name), f"common.__all__ lists {name!r} but it is not exported"


def test_key_extension_points_are_reexported():
    from common import (  # noqa: F401 - importability is the assertion
        ACTIVATION_REGISTRY,
        FUSION_REGISTRY,
        GROUPING_REGISTRY,
        METRIC_REGISTRY,
        FusionMethod,
        Grouping,
        Metric,
        Registry,
        StageContext,
    )

    assert "FusionMethod" in common.__all__
    assert FUSION_REGISTRY is common.FUSION_REGISTRY
