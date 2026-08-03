from dataclasses import replace

from moe_opt_core.config import load_experiment_config
from moe_opt_core.experiment import ExperimentStageId, TraceSource, build_experiment_plan


def test_stage_plan_is_stable_and_ends_with_artifacts() -> None:
    config = load_experiment_config("configs/experiments/h123_refactor_baseline.json")
    first = build_experiment_plan(config)
    second = build_experiment_plan(config)
    assert first == second
    assert first.stages[-1] == ExperimentStageId.ARTIFACTS
    assert ExperimentStageId.H1_DIAGNOSTICS in first.stages


def test_hf_dataset_source_adds_trace_stage() -> None:
    config = load_experiment_config("configs/experiments/h123_refactor_baseline.json")
    calibration = replace(config.calibration, source=TraceSource.HF_DATASET.value, dataset_name="dummy")
    plan = build_experiment_plan(replace(config, calibration=calibration))
    assert ExperimentStageId.COLLECT_TRACES in plan.stages


def test_config_normalization_is_idempotent() -> None:
    config = load_experiment_config("configs/experiments/h123_refactor_baseline.json")
    assert config.calibration.source == TraceSource.SYNTHETIC.value
    assert config.pilot.compression_fractions == (0.1, 0.25, 0.4)
