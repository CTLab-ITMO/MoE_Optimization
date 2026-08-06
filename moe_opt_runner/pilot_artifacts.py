from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import Iterable, Mapping

from moe_opt_core.experiment import PilotExecutionStateContract, PilotStepResultContract
from moe_opt_core.experiment.errors import ExpectedExperimentFailure, ExperimentFailureCode


def read_incremental_records(path: str | Path) -> tuple[Mapping[str, object], ...]:
    source = Path(path)
    if not source.exists():
        return ()
    lines = source.read_text(encoding="utf-8").splitlines()
    records: list[Mapping[str, object]] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break
            raise
        if not isinstance(value, Mapping):
            raise ValueError(f"Pilot ledger row {index + 1} is not an object")
        records.append(value)
    return tuple(records)


def write_json_atomic(path: str | Path, payload: Mapping[str, object]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(dict(payload), stream, indent=2, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(output)
    return output


def write_pilot_state(
    path: str | Path,
    state: PilotExecutionStateContract,
    result: PilotStepResultContract,
) -> Path:
    state_payload = asdict(state)
    state_payload["completed_steps"] = [step.value for step in state.completed_steps]
    result_payload = asdict(result)
    result_payload["step"] = result.step.value
    result_payload["status"] = result.status.value
    return write_json_atomic(
        path,
        {
            **state_payload,
            "last_result": result_payload,
        },
    )


def read_pilot_state_config_hash(path: str | Path) -> str | None:
    source = Path(path)
    if not source.exists():
        return None
    payload = json.loads(source.read_text(encoding="utf-8"))
    value = payload.get("config_hash")
    return str(value) if value is not None else None


def write_selection(
    path: str | Path,
    *,
    config_hash: str,
    candidate_ids: Iterable[str],
) -> Path:
    return write_json_atomic(
        path,
        {
            "schema_version": "1.0",
            "config_hash": config_hash,
            "selected_candidate_ids": list(candidate_ids),
        },
    )


def read_selection(path: str | Path, *, config_hash: str) -> tuple[str, ...]:
    source = Path(path)
    if not source.exists():
        return ()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("config_hash") != config_hash:
        raise ExpectedExperimentFailure(
            ExperimentFailureCode.PILOT_STATE_MISMATCH,
            "Pilot selection belongs to another experiment config",
            context={
                "expected_config_hash": config_hash,
                "observed_config_hash": payload.get("config_hash"),
            },
        )
    return tuple(str(value) for value in payload.get("selected_candidate_ids", ()))
