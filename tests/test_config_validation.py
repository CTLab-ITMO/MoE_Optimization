from pathlib import Path

from moe_opt_core.config import load_experiment_config


def test_load_example_config() -> None:
    cfg = load_experiment_config("configs/experiments/h123_refactor_baseline.json")
    assert cfg.name == "h123_refactor_baseline"
    assert cfg.runtime_top_k == 4
    assert cfg.runtime_expected_routed_experts == 60
    assert cfg.runtime_expected_shared_experts == 4
    assert cfg.runtime_strict_architecture is True
    assert "training" in cfg.constants


def test_missing_config_file_raises() -> None:
    missing = Path("configs/experiments/does_not_exist.json")
    try:
        load_experiment_config(missing)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError")
