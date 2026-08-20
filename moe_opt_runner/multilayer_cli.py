from __future__ import annotations

import argparse
import json

from moe_opt_core.config import load_experiment_config
from moe_opt_core.experiment import MultiLayerStep, PilotStepStatus
from moe_opt_core.experiment.protocol import materialize_replication_config
from moe_opt_runner.multilayer_executor import (
    ResumableMultiLayerExecutor,
    multilayer_step_result_dict,
)


TRACE_STEPS = {
    MultiLayerStep.TRACES_CALIBRATION: "calibration",
    MultiLayerStep.TRACES_VALIDATION: "validation",
    MultiLayerStep.TRACES_TEST: "test",
}


def _runtime_factory(config):
    from moe_opt_runner.runtime import build_runtime_context

    return lambda: build_runtime_context(config)


def execute_multilayer_step(
    config_path: str,
    step: MultiLayerStep,
    *,
    max_items: int | None = None,
    max_step_seconds: float | None = None,
    retry_failed: bool = False,
    replication_seed: int | None = None,
) -> dict[str, object]:
    config = materialize_replication_config(
        load_experiment_config(config_path),
        replication_seed,
    )
    executor = ResumableMultiLayerExecutor(config)
    item_limit = (
        max_items
        if max_items is not None
        else config.multilayer.candidates_per_invocation
    )
    time_limit = (
        max_step_seconds
        if max_step_seconds is not None
        else config.multilayer.max_step_seconds
    )

    if step in TRACE_STEPS:
        split = TRACE_STEPS[step]
        if split == "test" and not executor.selection_path.exists():
            raise RuntimeError("Test traces are locked until configuration selection")
        pending_layers = executor.pending_trace_layers(split)
        if not pending_layers:
            result = executor._result(
                step,
                PilotStepStatus.SKIPPED,
                0,
                0,
                executor._trace_paths(split),
            )
        else:
            ctx = _runtime_factory(config)()
            selected_layers = pending_layers[:item_limit]
            batch_path = executor._batch_path(split)
            if batch_path.exists():
                import torch

                from moe_opt_runner.tracing import collect_traces_from_batches

                batches = torch.load(batch_path, map_location="cpu")
                collect_traces_from_batches(
                    config,
                    ctx,
                    {split: batches},
                    layers=selected_layers,
                )
            else:
                from moe_opt_runner.tracing import collect_real_calibration_traces

                collect_real_calibration_traces(
                    config,
                    ctx,
                    splits=(split,),
                    layers=selected_layers,
                )
            result = executor.record_trace_step(
                split,
                processed_items=len(selected_layers),
            )
    elif step == MultiLayerStep.LAYER_DIAGNOSTICS:
        executor.ctx = _runtime_factory(config)()
        result = executor.run_layer_diagnostics_batch(
            max_items=item_limit,
            max_seconds=time_limit,
        )
    elif step in {
        MultiLayerStep.BASELINE_VALIDATION,
        MultiLayerStep.SENSITIVITY_VALIDATION,
        MultiLayerStep.MULTILAYER_VALIDATION,
        MultiLayerStep.BASELINE_TEST,
        MultiLayerStep.TEST_SELECTED,
    }:
        executor.ctx = _runtime_factory(config)()
        if step == MultiLayerStep.BASELINE_VALIDATION:
            result = executor.run_baseline("validation")
        elif step == MultiLayerStep.SENSITIVITY_VALIDATION:
            result = executor.run_sensitivity_batch(
                max_items=item_limit,
                max_seconds=time_limit,
                retry_failed=retry_failed,
            )
        elif step == MultiLayerStep.MULTILAYER_VALIDATION:
            result = executor.run_recipe_batch(
                split="validation",
                max_items=item_limit,
                max_seconds=time_limit,
                retry_failed=retry_failed,
            )
        elif step == MultiLayerStep.BASELINE_TEST:
            result = executor.run_baseline("test")
        else:
            result = executor.run_recipe_batch(
                split="test",
                max_items=item_limit,
                max_seconds=time_limit,
                retry_failed=retry_failed,
            )
    elif step == MultiLayerStep.BUILD_RECIPES:
        result = executor.build_recipes()
    elif step == MultiLayerStep.SELECT_CONFIGURATIONS:
        result = executor.select_configurations()
    elif step == MultiLayerStep.MATERIALIZE:
        result = executor.materialize_batch(
            _runtime_factory(config),
            max_items=item_limit,
        )
    elif step == MultiLayerStep.BENCHMARK_MATERIALIZED:
        result = executor.benchmark_materialized_batch(
            _runtime_factory(config),
            max_items=item_limit,
        )
    elif step == MultiLayerStep.BUILD_SUMMARY:
        result = executor.build_summary()
    else:
        raise ValueError(f"Unsupported multi-layer step: {step.value}")
    return multilayer_step_result_dict(result)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one bounded, resumable multi-layer MoE compression step"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--step",
        required=True,
        choices=("status", *(step.value for step in MultiLayerStep)),
    )
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--max-step-seconds", type=float, default=None)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--replication-seed", type=int, default=None)
    args = parser.parse_args()
    if args.max_items is not None and args.max_items < 1:
        parser.error("--max-items must be positive")
    if args.max_step_seconds is not None and args.max_step_seconds <= 0.0:
        parser.error("--max-step-seconds must be positive")

    if args.step == "status":
        config = materialize_replication_config(
            load_experiment_config(args.config),
            args.replication_seed,
        )
        output = ResumableMultiLayerExecutor(config).status()
    else:
        output = execute_multilayer_step(
            args.config,
            MultiLayerStep(args.step),
            max_items=args.max_items,
            max_step_seconds=args.max_step_seconds,
            retry_failed=args.retry_failed,
            replication_seed=args.replication_seed,
        )
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
