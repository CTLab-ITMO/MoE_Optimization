from dataclasses import replace
from pathlib import Path

import pytest

from moe_opt_core.config import load_experiment_config
from moe_opt_core.experiment import (
    CandidateAction,
    CandidateMethod,
    CandidatePlanContract,
    EvaluationMetricsContract,
    PilotStep,
    PilotStepStatus,
)
from moe_opt_core.experiment.errors import ExpectedExperimentFailure, ExperimentFailureCode
from moe_opt_core.experiment.pilot_planning import (
    build_pilot_execution_state,
    next_pilot_step,
    pending_candidate_ids,
    select_validation_winners,
)
from moe_opt_runner.evaluation import append_incremental_record
from moe_opt_runner.pilot_artifacts import (
    read_incremental_records,
    write_json_atomic,
    write_selection,
)
from moe_opt_runner.pilot_cli import execute_step
from moe_opt_runner.pilot_executor import ResumablePilotExecutor


def _plan(candidate_id: str, fraction: float = 0.25) -> CandidatePlanContract:
    return CandidatePlanContract(
        candidate_id=candidate_id,
        layer=0,
        method=CandidateMethod.RMT_GEOMETRY,
        action=CandidateAction.MERGE,
        compression_fraction=fraction,
        groups=((0, 1), (2,), (3,)),
        seed=1,
    )


def _metrics(perplexity: float = 2.0) -> dict[str, object]:
    return {
        "nll": 0.5,
        "perplexity": perplexity,
        "tokens_per_second": 10.0,
        "peak_memory_mb": 1.0,
        "model_parameter_mb": 2.0,
        "evaluated_tokens": 8,
        "elapsed_seconds": 0.1,
    }


def _config(tmp_path: Path):
    config = load_experiment_config("configs/experiments/h123_refactor_baseline.json")
    artifacts = replace(
        config.artifacts,
        root_dir=str(tmp_path / "artifacts"),
        manifest_path=str(tmp_path / "artifacts" / "manifest.json"),
    )
    calibration = replace(config.calibration, output_dir=str(tmp_path / "traces"))
    h1 = replace(
        config.h1,
        mlp_inputs_path=str(tmp_path / "traces" / "h1_inputs.pt"),
        output_pairs_path=str(tmp_path / "artifacts" / "pairs_0.pt"),
        output_diagnostics_path=str(tmp_path / "artifacts" / "h1_diag_0.json"),
        output_merged_dir=str(tmp_path / "artifacts" / "merged"),
    )
    pilot = replace(config.pilot, enabled=True)
    return replace(config, artifacts=artifacts, calibration=calibration, h1=h1, pilot=pilot)


def test_pending_candidates_and_winner_selection_are_deterministic() -> None:
    records = (
        {
            "config_hash": "hash",
            "record_type": "candidate",
            "split": "validation",
            "plan": {"candidate_id": "a", "method": "rmt_geometry", "compression_fraction": 0.25},
            "metrics": {"perplexity": 3.0},
            "relative_perplexity_degradation": 0.02,
        },
        {
            "config_hash": "hash",
            "record_type": "candidate",
            "split": "validation",
            "plan": {"candidate_id": "b", "method": "rmt_geometry", "compression_fraction": 0.25},
            "metrics": {"perplexity": 2.0},
            "relative_perplexity_degradation": 0.01,
        },
        {
            "config_hash": "hash",
            "record_type": "failed_candidate",
            "split": "validation",
            "plan": {"candidate_id": "c"},
        },
    )
    state = build_pilot_execution_state(
        config_hash="hash",
        records=records,
        all_validation_candidate_ids=("a", "b", "c"),
        selected_candidate_ids=(),
    )
    assert PilotStep.VALIDATE_CANDIDATES in state.completed_steps
    assert pending_candidate_ids(("a", "b", "c"), state, split="validation", retry_failed=False) == ()
    assert pending_candidate_ids(("a", "b", "c"), state, split="validation", retry_failed=True) == ("c",)
    assert select_validation_winners(
        records=records,
        config_hash="hash",
        candidate_order=("a", "b", "c"),
        max_quality_budget=0.05,
    ) == ("b",)


def test_robust_winner_selection_penalizes_unstable_candidates() -> None:
    def row(candidate_id: str, perplexity: float, score: float, passes: bool) -> dict:
        return {
            "config_hash": "hash",
            "record_type": "candidate",
            "split": "validation",
            "plan": {
                "candidate_id": candidate_id,
                "method": "rmt_geometry",
                "compression_fraction": 0.25,
            },
            "metrics": {"perplexity": perplexity},
            "relative_perplexity_degradation": perplexity / 2.0 - 1.0,
            "validation_robustness": {
                "robust_score": score,
                "passes_max_slice_degradation": passes,
            },
        }

    records = (
        row("lower_aggregate", 2.02, 0.04, True),
        row("lower_dispersion", 2.03, 0.02, True),
        row("hard_limit_failed", 2.00, 0.0, False),
    )
    assert select_validation_winners(
        records=records,
        config_hash="hash",
        candidate_order=("lower_aggregate", "lower_dispersion", "hard_limit_failed"),
        max_quality_budget=0.05,
    ) == ("lower_dispersion",)


def test_pilot_requires_validation_traces_before_h1_diagnostics() -> None:
    assert next_pilot_step(
        (PilotStep.TRACES_CALIBRATION,),
        remaining_validation_candidates=0,
        remaining_test_candidates=0,
    ) == PilotStep.TRACES_VALIDATION
    assert next_pilot_step(
        (PilotStep.TRACES_CALIBRATION, PilotStep.TRACES_VALIDATION),
        remaining_validation_candidates=0,
        remaining_test_candidates=0,
    ) == PilotStep.H1_DIAGNOSTICS


def test_h1_step_rejects_missing_validation_trace(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr("moe_opt_runner.pilot_cli.load_experiment_config", lambda _path: config)
    with pytest.raises(ExpectedExperimentFailure, match="held-out spectral stability") as failure:
        execute_step("unused.json", PilotStep.H1_DIAGNOSTICS)
    assert failure.value.code == ExperimentFailureCode.PILOT_STATE_MISMATCH
    assert failure.value.context["required_step"] == PilotStep.TRACES_VALIDATION.value


def test_candidate_batches_resume_without_duplicate_completed_rows(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path)
    plans = (_plan("a"), _plan("b", 0.3), _plan("c", 0.4))
    batch_path = Path(config.calibration.output_dir) / "validation" / "token_batches.pt"
    batch_path.parent.mkdir(parents=True)
    batch_path.write_bytes(b"placeholder")

    executor = ResumablePilotExecutor(config)
    append_incremental_record(
        executor.record_path,
        {
            "config_hash": executor.run_hash,
            "record_type": "baseline",
            "split": "validation",
            "metrics": _metrics(),
        },
    )
    monkeypatch.setattr("moe_opt_runner.pilot_executor.torch.load", lambda *args, **kwargs: [object()])

    def fake_evaluate(plan, batches, baseline, expert_cache, *, split):
        assert split == "validation"
        return {
            "config_hash": executor.run_hash,
            "plan": {
                "candidate_id": plan.candidate_id,
                "method": plan.method.value,
                "compression_fraction": plan.compression_fraction,
            },
            "metrics": _metrics(),
            "relative_perplexity_degradation": 0.0,
            "throughput_gain": 0.0,
            "memory_reduction": 0.0,
            "accepted_quality_budgets": {"0.05": True},
        }

    monkeypatch.setattr(executor, "_evaluate_plan", fake_evaluate)
    first = executor._run_candidate_batch(
        split="validation",
        plans=plans,
        max_items=2,
        max_seconds=60.0,
        retry_failed=False,
    )
    assert first.status == PilotStepStatus.PARTIAL
    assert first.processed_items == 2
    assert first.remaining_items == 1
    executor._trace_path().parent.mkdir(parents=True, exist_ok=True)
    executor._trace_path().write_bytes(b"trace")
    executor._embedding_path().parent.mkdir(parents=True, exist_ok=True)
    executor._embedding_path().write_bytes(b"embedding")
    monkeypatch.setattr(executor, "_load_plans_without_runtime", lambda: plans)
    status = executor.status()
    assert status["total_validation_candidates"] == 3
    assert status["remaining_validation_candidates"] == 1

    resumed = ResumablePilotExecutor(config)
    monkeypatch.setattr(resumed, "_evaluate_plan", fake_evaluate)
    second = resumed._run_candidate_batch(
        split="validation",
        plans=plans,
        max_items=2,
        max_seconds=60.0,
        retry_failed=False,
    )
    assert second.status == PilotStepStatus.COMPLETED
    assert second.processed_items == 1
    assert second.remaining_items == 0

    third = resumed._run_candidate_batch(
        split="validation",
        plans=plans,
        max_items=2,
        max_seconds=60.0,
        retry_failed=False,
    )
    assert third.status == PilotStepStatus.SKIPPED
    candidate_rows = [
        row
        for row in read_incremental_records(executor.record_path)
        if row.get("record_type") == "candidate"
    ]
    assert [row["plan"]["candidate_id"] for row in candidate_rows] == ["a", "b", "c"]


def test_reader_ignores_only_a_truncated_final_ledger_row(tmp_path: Path) -> None:
    ledger = tmp_path / "pilot_runs.jsonl"
    ledger.write_text('{"record_type": "baseline"}\n{"record_type":', encoding="utf-8")
    assert read_incremental_records(ledger) == ({"record_type": "baseline"},)


def test_multislice_baseline_is_persisted_with_aggregate_metrics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _config(tmp_path)
    validation_selection = replace(
        config.pilot.validation_selection,
        slice_count=3,
        std_penalty=1.0,
        max_slice_degradation=0.05,
    )
    config = replace(
        config,
        pilot=replace(
            config.pilot,
            max_eval_batches=7,
            validation_selection=validation_selection,
        ),
        profile=replace(config.profile, measure_steps=7),
    )
    batch_path = Path(config.calibration.output_dir) / "validation" / "token_batches.pt"
    batch_path.parent.mkdir(parents=True)
    batch_path.write_bytes(b"placeholder")
    monkeypatch.setattr(
        "moe_opt_runner.pilot_executor.torch.load",
        lambda *args, **kwargs: tuple(range(7)),
    )

    executor = ResumablePilotExecutor(config)
    observed_slices = []

    def fake_evaluate(batches, *, warmup_steps=None, measure_steps=None):
        observed_slices.append(tuple(batches))
        nll = 1.0 + 0.1 * len(observed_slices)
        return EvaluationMetricsContract(
            nll=nll,
            perplexity=2.0,
            tokens_per_second=10.0,
            peak_memory_mb=100.0,
            model_parameter_mb=200.0,
            evaluated_tokens=10 * len(batches),
            elapsed_seconds=float(len(batches)),
        )

    monkeypatch.setattr(executor, "_evaluate", fake_evaluate)
    result = executor.run_baseline("validation")

    assert result.status == PilotStepStatus.COMPLETED
    assert observed_slices == [(0, 3, 6), (1, 4), (2, 5)]
    row = read_incremental_records(executor.record_path)[0]
    assert row["validation_slices"]["slice_count"] == 3
    assert len(row["validation_slices"]["metrics"]) == 3
    assert row["metrics"]["evaluated_tokens"] == 70


def test_multislice_validation_reports_too_few_batches_as_expected_failure(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config = replace(
        config,
        pilot=replace(
            config.pilot,
            max_eval_batches=3,
            validation_selection=replace(
                config.pilot.validation_selection,
                slice_count=3,
            ),
        ),
        profile=replace(config.profile, measure_steps=3),
    )
    executor = ResumablePilotExecutor(config)
    with pytest.raises(ExpectedExperimentFailure) as failure:
        executor._validation_batch_slices((object(), object()))
    assert failure.value.code == ExperimentFailureCode.EVALUATION_FAILED
    assert failure.value.context == {"available_batches": 2, "slice_count": 3}


def test_executor_rejects_state_from_another_config(tmp_path: Path) -> None:
    executor = ResumablePilotExecutor(_config(tmp_path))
    write_json_atomic(executor.state_path, {"config_hash": "another-run"})
    with pytest.raises(ExpectedExperimentFailure) as failure:
        executor.status()
    assert failure.value.code == ExperimentFailureCode.PILOT_STATE_MISMATCH


def test_selection_locks_validation_and_test_is_locked_before_selection(tmp_path: Path) -> None:
    executor = ResumablePilotExecutor(_config(tmp_path))
    with pytest.raises(ExpectedExperimentFailure, match="Test baseline is locked"):
        executor.run_baseline("test")
    write_selection(executor.selection_path, config_hash=executor.run_hash, candidate_ids=())
    with pytest.raises(ExpectedExperimentFailure, match="Validation is locked"):
        executor.run_validation_batch(max_items=1, max_seconds=1.0)
