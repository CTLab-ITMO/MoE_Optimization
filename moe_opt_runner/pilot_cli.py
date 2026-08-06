from __future__ import annotations

import argparse
import json
from pathlib import Path

from moe_opt_core.config import load_experiment_config
from moe_opt_core.experiment import PilotStep
from moe_opt_core.experiment.errors import ExpectedExperimentFailure, ExperimentFailureCode
from moe_opt_runner.pilot_executor import ResumablePilotExecutor, pilot_step_result_dict


TRACE_STEPS = {
    PilotStep.TRACES_CALIBRATION: "calibration",
    PilotStep.TRACES_VALIDATION: "validation",
    PilotStep.TRACES_TEST: "test",
}


def _trace_artifact_paths(config, split: str) -> tuple[Path, ...]:
    root = Path(config.calibration.output_dir)
    return (
        root / "trace_manifest.json",
        root / split / f"layer_{config.pilot.layer}.pt",
        root / split / "token_batches.pt",
    )


def _h1_artifact_paths(config) -> tuple[Path, ...]:
    diagnostics = Path(config.h1.output_diagnostics_path)
    return (
        diagnostics,
        diagnostics.with_suffix(".pt"),
        Path(config.h1.output_pairs_path),
    )


def _executor_with_runtime(config):
    from moe_opt_runner.runtime import build_runtime_context

    return ResumablePilotExecutor(config, build_runtime_context(config))


def execute_step(
    config_path: str,
    step: PilotStep,
    *,
    max_items: int | None = None,
    max_step_seconds: float | None = None,
    retry_failed: bool = False,
) -> dict[str, object]:
    config = load_experiment_config(config_path)
    if not config.pilot.enabled:
        raise ValueError("The resumable pilot runner requires pilot.enabled=true")
    executor = ResumablePilotExecutor(config)

    if step in TRACE_STEPS:
        split = TRACE_STEPS[step]
        if split == "test" and not executor.has_selection():
            raise ExpectedExperimentFailure(
                ExperimentFailureCode.PILOT_STATE_MISMATCH,
                "Test traces are locked until select_candidates is completed",
            )
        paths = _trace_artifact_paths(config, split)
        if executor.is_step_complete(step):
            return pilot_step_result_dict(executor.skip_step(step, paths))
        executor.begin_step(step)
        executor = _executor_with_runtime(config)
        from moe_opt_runner.tracing import collect_real_calibration_traces

        collect_real_calibration_traces(config, executor.ctx, splits=(split,))
        return pilot_step_result_dict(executor.record_artifact_step(step, paths))

    if step == PilotStep.H1_DIAGNOSTICS:
        paths = _h1_artifact_paths(config)
        if executor.is_step_complete(step):
            return pilot_step_result_dict(executor.skip_step(step, paths))
        executor.begin_step(step)
        executor = _executor_with_runtime(config)
        from moe_opt_runner.stages import run_h1_stage

        run_h1_stage(config, ctx=executor.ctx)
        return pilot_step_result_dict(executor.record_artifact_step(step, paths))

    if step in {
        PilotStep.BASELINE_VALIDATION,
        PilotStep.BASELINE_TEST,
        PilotStep.VALIDATE_CANDIDATES,
        PilotStep.TEST_SELECTED,
    }:
        executor.begin_step(step)
        executor = _executor_with_runtime(config)

    if step in {PilotStep.SELECT_CANDIDATES, PilotStep.BUILD_SUMMARY}:
        executor.begin_step(step)

    if step == PilotStep.BASELINE_VALIDATION:
        result = executor.run_baseline("validation")
    elif step == PilotStep.BASELINE_TEST:
        result = executor.run_baseline("test")
    elif step == PilotStep.VALIDATE_CANDIDATES:
        result = executor.run_validation_batch(
            max_items=(
                max_items if max_items is not None else config.pilot.candidates_per_invocation
            ),
            max_seconds=(
                max_step_seconds if max_step_seconds is not None else config.pilot.max_step_seconds
            ),
            retry_failed=retry_failed,
        )
    elif step == PilotStep.SELECT_CANDIDATES:
        result = executor.select_candidates()
    elif step == PilotStep.TEST_SELECTED:
        result = executor.run_test_batch(
            max_items=(
                max_items if max_items is not None else config.pilot.candidates_per_invocation
            ),
            max_seconds=(
                max_step_seconds if max_step_seconds is not None else config.pilot.max_step_seconds
            ),
            retry_failed=retry_failed,
        )
    elif step == PilotStep.BUILD_SUMMARY:
        result = executor.build_summary()
    else:
        raise ValueError(f"Unsupported pilot step: {step.value}")
    return pilot_step_result_dict(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one bounded, resumable Qwen-MoE pilot step")
    parser.add_argument("--config", required=True, help="Experiment JSON config")
    parser.add_argument(
        "--step",
        required=True,
        choices=("status", *(step.value for step in PilotStep)),
    )
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--max-step-seconds", type=float, default=None)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()

    if args.max_items is not None and args.max_items < 1:
        parser.error("--max-items must be positive")
    if args.max_step_seconds is not None and args.max_step_seconds <= 0.0:
        parser.error("--max-step-seconds must be positive")

    if args.step == "status":
        config = load_experiment_config(args.config)
        output = ResumablePilotExecutor(config).status()
    else:
        output = execute_step(
            args.config,
            PilotStep(args.step),
            max_items=args.max_items,
            max_step_seconds=args.max_step_seconds,
            retry_failed=args.retry_failed,
        )
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
