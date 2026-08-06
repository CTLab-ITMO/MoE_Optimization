from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from moe_opt_core.experiment.contracts import PilotExecutionStateContract, PilotStep


def next_pilot_step(
    completed_steps: Sequence[PilotStep],
    *,
    remaining_validation_candidates: int,
    remaining_test_candidates: int,
) -> PilotStep | None:
    completed = set(completed_steps)
    if PilotStep.TRACES_CALIBRATION not in completed:
        return PilotStep.TRACES_CALIBRATION
    if PilotStep.TRACES_VALIDATION not in completed:
        return PilotStep.TRACES_VALIDATION
    if PilotStep.H1_DIAGNOSTICS not in completed:
        return PilotStep.H1_DIAGNOSTICS
    if PilotStep.BASELINE_VALIDATION not in completed:
        return PilotStep.BASELINE_VALIDATION
    if remaining_validation_candidates:
        return PilotStep.VALIDATE_CANDIDATES
    if PilotStep.SELECT_CANDIDATES not in completed:
        return PilotStep.SELECT_CANDIDATES
    if PilotStep.TRACES_TEST not in completed:
        return PilotStep.TRACES_TEST
    if PilotStep.BASELINE_TEST not in completed:
        return PilotStep.BASELINE_TEST
    if remaining_test_candidates:
        return PilotStep.TEST_SELECTED
    if PilotStep.BUILD_SUMMARY not in completed:
        return PilotStep.BUILD_SUMMARY
    return None


def _matching_records(
    records: Iterable[Mapping[str, object]],
    config_hash: str,
) -> tuple[Mapping[str, object], ...]:
    return tuple(row for row in records if row.get("config_hash") == config_hash)


def _candidate_id(row: Mapping[str, object]) -> str | None:
    plan = row.get("plan")
    if not isinstance(plan, Mapping):
        return None
    value = plan.get("candidate_id")
    return str(value) if value is not None else None


def _candidate_ids(
    records: Sequence[Mapping[str, object]],
    record_type: str,
) -> tuple[str, ...]:
    values = {
        candidate_id
        for row in records
        if row.get("record_type") == record_type
        for candidate_id in (_candidate_id(row),)
        if candidate_id is not None
    }
    return tuple(sorted(values))


def build_pilot_execution_state(
    *,
    config_hash: str,
    records: Iterable[Mapping[str, object]],
    all_validation_candidate_ids: Sequence[str],
    selected_candidate_ids: Sequence[str],
    artifact_steps: Sequence[PilotStep] = (),
    selection_written: bool = False,
    summary_written: bool = False,
) -> PilotExecutionStateContract:
    matching = _matching_records(records, config_hash)
    completed_validation = _candidate_ids(matching, "candidate")
    failed_validation = _candidate_ids(matching, "failed_candidate")
    completed_test = _candidate_ids(matching, "selected_candidate")
    failed_test = _candidate_ids(matching, "failed_selected_candidate")

    completed_steps = set(artifact_steps)
    if any(row.get("record_type") == "baseline" and row.get("split") == "validation" for row in matching):
        completed_steps.add(PilotStep.BASELINE_VALIDATION)
    if any(row.get("record_type") == "baseline" and row.get("split") == "test" for row in matching):
        completed_steps.add(PilotStep.BASELINE_TEST)

    validation_terminal = set(completed_validation) | set(failed_validation)
    if all_validation_candidate_ids and set(all_validation_candidate_ids) <= validation_terminal:
        completed_steps.add(PilotStep.VALIDATE_CANDIDATES)
    if selection_written:
        completed_steps.add(PilotStep.SELECT_CANDIDATES)

    test_terminal = set(completed_test) | set(failed_test)
    if selection_written and set(selected_candidate_ids) <= test_terminal:
        completed_steps.add(PilotStep.TEST_SELECTED)
    if summary_written:
        completed_steps.add(PilotStep.BUILD_SUMMARY)

    ordered_steps = tuple(step for step in PilotStep if step in completed_steps)
    return PilotExecutionStateContract(
        schema_version="1.0",
        config_hash=config_hash,
        completed_steps=ordered_steps,
        completed_validation_candidate_ids=completed_validation,
        failed_validation_candidate_ids=failed_validation,
        selected_candidate_ids=tuple(selected_candidate_ids),
        completed_test_candidate_ids=completed_test,
        failed_test_candidate_ids=failed_test,
    )


def pending_candidate_ids(
    candidate_ids: Sequence[str],
    state: PilotExecutionStateContract,
    *,
    split: str,
    retry_failed: bool,
) -> tuple[str, ...]:
    if split == "validation":
        completed = set(state.completed_validation_candidate_ids)
        failed = set(state.failed_validation_candidate_ids)
    elif split == "test":
        completed = set(state.completed_test_candidate_ids)
        failed = set(state.failed_test_candidate_ids)
    else:
        raise ValueError("split must be 'validation' or 'test'")
    terminal = completed if retry_failed else completed | failed
    return tuple(candidate_id for candidate_id in candidate_ids if candidate_id not in terminal)


def select_validation_winners(
    *,
    records: Iterable[Mapping[str, object]],
    config_hash: str,
    candidate_order: Sequence[str],
    max_quality_budget: float,
) -> tuple[str, ...]:
    latest_rows: dict[str, Mapping[str, object]] = {}
    for row in _matching_records(records, config_hash):
        if row.get("record_type") != "candidate" or row.get("split") != "validation":
            continue
        candidate_id = _candidate_id(row)
        if candidate_id is not None:
            latest_rows[candidate_id] = row

    winners: dict[tuple[str, float], tuple[str, float]] = {}
    for candidate_id in candidate_order:
        row = latest_rows.get(candidate_id)
        if row is None:
            continue
        degradation = float(row.get("relative_perplexity_degradation", float("inf")))
        if degradation > max_quality_budget:
            continue
        plan = row.get("plan")
        metrics = row.get("metrics")
        if not isinstance(plan, Mapping) or not isinstance(metrics, Mapping):
            continue
        key = (str(plan["method"]), float(plan["compression_fraction"]))
        perplexity = float(metrics["perplexity"])
        current = winners.get(key)
        if current is None or perplexity < current[1]:
            winners[key] = (candidate_id, perplexity)

    selected = {candidate_id for candidate_id, _ in winners.values()}
    return tuple(candidate_id for candidate_id in candidate_order if candidate_id in selected)
