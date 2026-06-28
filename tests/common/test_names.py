"""Tests for the named string contracts shared by producers and consumers."""

from common import names
from common.names import PairingMethod, ReplaceMode


def test_pairing_method_values_and_order_match_frozen_registry():
    # Order/set must match the frozen registry snapshot ``pairing_methods``.
    assert [str(m) for m in PairingMethod] == [
        "activation_correlation",
        "random",
        "output_similarity",
    ]


def test_pairing_method_compares_equal_to_plain_string():
    # config.PAIRING_METHOD is a plain string; StrEnum members must match it.
    assert PairingMethod.ACTIVATION_CORRELATION == "activation_correlation"
    assert "random" == PairingMethod.RANDOM


def test_replace_mode_tags_format_as_bare_values():
    # Stage-9 filenames interpolate these; f-string must yield the bare tag
    # (the golden harness reads ``accuracy_list_single_*``).
    assert f"{ReplaceMode.SINGLE}" == "single"
    assert f"{ReplaceMode.SEQ}" == "seq"
    assert f"{ReplaceMode.SMART_SEQ}" == "smart_seq"


def test_string_contract_constants_are_frozen():
    assert names.TIME_METRIC == "Time"
    assert names.BEST_PREFIX == "Best_"
    assert names.ZERO_FUSION == "zero"
    assert names.TEST_RESULT_PREFIX == "test_result_"
