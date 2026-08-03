from __future__ import annotations

from moe_opt_core.experiment.contracts import ExperimentPlan, ExperimentStageId, TraceSource


def build_experiment_plan(config) -> ExperimentPlan:
    stages: list[ExperimentStageId] = []
    has_algorithm_stage = config.h1.enabled or config.h2.enabled or config.h3.enabled
    if has_algorithm_stage or config.pilot.enabled:
        stages.append(ExperimentStageId.RUNTIME)

    if config.calibration.source == TraceSource.HF_DATASET.value:
        stages.append(ExperimentStageId.COLLECT_TRACES)

    if config.h1.enabled:
        stages.append(ExperimentStageId.H1_DIAGNOSTICS)
    if config.h2.enabled:
        stages.append(ExperimentStageId.H2_BIAS)
    if config.h3.enabled:
        stages.append(ExperimentStageId.H3_ROUTING)

    if config.pilot.enabled:
        stages.extend(
            (
                ExperimentStageId.BASELINE_EVALUATION,
                ExperimentStageId.CANDIDATE_EVALUATION,
            )
        )

    stages.append(ExperimentStageId.ARTIFACTS)
    return ExperimentPlan(experiment_name=config.name, stages=tuple(stages))
