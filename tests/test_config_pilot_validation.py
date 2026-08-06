import json
from pathlib import Path

import pytest

from moe_opt_core.config import load_experiment_config, validate_experiment_config_dict


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


def test_resumable_limits_must_be_positive() -> None:
    raw = _raw_config()
    raw["pilot"]["candidates_per_invocation"] = 0
    with pytest.raises(ValueError, match="candidates_per_invocation"):
        validate_experiment_config_dict(raw)
    raw = _raw_config()
    raw["pilot"]["max_step_seconds"] = 0
    with pytest.raises(ValueError, match="max_step_seconds"):
        validate_experiment_config_dict(raw)


def test_hybrid_pairing_weights_must_define_a_non_negative_score() -> None:
    raw = _raw_config()
    raw["pilot"]["hybrid_pairing"] = {
        "rmt_weight": 0.0,
        "functional_weight": 0.0,
        "usage_weight": 0.0,
    }
    with pytest.raises(ValueError, match="at least one positive"):
        validate_experiment_config_dict(raw)

    raw["pilot"]["hybrid_pairing"]["rmt_weight"] = -0.1
    with pytest.raises(ValueError, match="non-negative"):
        validate_experiment_config_dict(raw)


def test_spectral_pairing_rank_grid_must_be_positive_unique_and_bounded() -> None:
    raw = _raw_config()
    raw["pilot"]["spectral_pairing_ranks"] = [1, 0]
    with pytest.raises(ValueError, match="positive"):
        validate_experiment_config_dict(raw)
    raw["pilot"]["spectral_pairing_ranks"] = [1, 1]
    with pytest.raises(ValueError, match="unique"):
        validate_experiment_config_dict(raw)
    raw["pilot"]["spectral_pairing_ranks"] = [61]
    with pytest.raises(ValueError, match="expected_routed_experts"):
        validate_experiment_config_dict(raw)


def test_hybrid_weight_variants_require_unique_names_and_valid_weights() -> None:
    raw = _raw_config()
    raw["pilot"]["hybrid_pairing"] = {
        "variants": [
            {
                "name": "same",
                "rmt_weight": 0.5,
                "functional_weight": 0.3,
                "usage_weight": 0.2,
            },
            {
                "name": "same",
                "rmt_weight": 0.2,
                "functional_weight": 0.3,
                "usage_weight": 0.5,
            },
        ]
    }
    with pytest.raises(ValueError, match="names must be unique"):
        validate_experiment_config_dict(raw)
    raw["pilot"]["hybrid_pairing"]["variants"][1]["name"] = "invalid"
    raw["pilot"]["hybrid_pairing"]["variants"][1]["usage_weight"] = -0.5
    with pytest.raises(ValueError, match="variant weights must be non-negative"):
        validate_experiment_config_dict(raw)


def test_new_pairing_method_is_opt_in_for_legacy_pilot_defaults(tmp_path) -> None:
    raw = _raw_config()
    raw["pilot"].pop("methods")
    path = tmp_path / "config.json"
    source = Path("configs/experiments/h1_rmt_one_layer_pilot.json").resolve()
    raw["constants"] = {
        key: str((source.parent / value).resolve())
        for key, value in raw["constants"].items()
    }
    path.write_text(json.dumps(raw), encoding="utf-8")
    config = load_experiment_config(path)
    assert "usage_guided_merge" not in config.pilot.methods
