import json

import pytest

from moe_opt_core.config import validate_experiment_config_dict


def _raw_config() -> dict:
    with open("configs/experiments/h1_rmt_one_layer_pilot.json", encoding="utf-8") as stream:
        return json.load(stream)


def test_real_dataset_requires_dataset_name() -> None:
    raw = _raw_config()
    raw["calibration"]["dataset_name"] = ""
    with pytest.raises(ValueError, match="dataset_name"):
        validate_experiment_config_dict(raw)


def test_pilot_layer_must_be_in_experiment_layers() -> None:
    raw = _raw_config()
    raw["pilot"]["layer"] = 3
    with pytest.raises(ValueError, match="pilot.layer"):
        validate_experiment_config_dict(raw)


def test_runtime_top_k_is_bounded_by_expected_routed_experts() -> None:
    raw = _raw_config()
    raw["runtime"]["top_k"] = 61
    with pytest.raises(ValueError, match="expected_routed_experts"):
        validate_experiment_config_dict(raw)


def test_pilot_rejects_synthetic_traces_when_real_traces_are_required() -> None:
    raw = _raw_config()
    raw["calibration"]["source"] = "synthetic"
    with pytest.raises(ValueError, match="require_real_traces"):
        validate_experiment_config_dict(raw)
