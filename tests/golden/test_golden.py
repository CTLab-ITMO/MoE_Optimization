"""Golden gate: the fusion pipeline reproduces the frozen ``golden_v1.json``.

Captured once on the pristine legacy module (``UPDATE_GOLDEN=1``) and then held
as a checked-in regression oracle. Both the legacy module and the new
``experts_fusion`` copy must reproduce it within the tight/loose tolerance
policy (``tolerances.py``); a self-consistency test proves the capture is
deterministic (double regeneration is identical).

Re-basing the golden is intentional and rare: run with ``UPDATE_GOLDEN=1`` and
review the JSON diff in the PR. The plan sanctions exactly two re-bases (S4
local seeds, S5 ``Best_*`` fallback); any other change to values / orientation /
order is a red gate.
"""
import json
import os
from pathlib import Path

import pytest

import harness
import tolerances

GOLDEN_PATH = Path(__file__).resolve().parent / "golden_v1.json"
UPDATE = bool(os.environ.get("UPDATE_GOLDEN"))


def _load_golden():
    if not GOLDEN_PATH.exists():
        return None
    with open(GOLDEN_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, allow_nan=True)


def _require_golden(golden):
    if golden is None:
        pytest.skip("golden_v1.json missing; run pytest with UPDATE_GOLDEN=1 to create it")


def test_registry_snapshot_matches_golden():
    golden = _load_golden()
    _require_golden(golden)
    problems = tolerances.diff_registry(harness.registry_snapshot(), golden["registry"])
    assert not problems, "registry snapshot drift:\n" + "\n".join(problems)


def test_golden_legacy_module(tmp_path, monkeypatch):
    aggregates = harness.run_chain(harness.LEGACY, tmp_path, monkeypatch)

    if UPDATE:
        snapshot = {"registry": harness.registry_snapshot(), **aggregates}
        with open(GOLDEN_PATH, "w", encoding="utf-8") as fh:
            json.dump(snapshot, fh, indent=2, sort_keys=True, allow_nan=True)
        return

    golden = _load_golden()
    _require_golden(golden)
    problems = tolerances.diff_aggregates(aggregates, golden)
    assert not problems, "legacy module drift vs golden_v1:\n" + "\n".join(problems)


def test_golden_new_module(tmp_path, monkeypatch):
    golden = _load_golden()
    _require_golden(golden)
    aggregates = harness.run_chain(harness.NEW, tmp_path, monkeypatch)
    problems = tolerances.diff_aggregates(aggregates, golden)
    assert not problems, "experts_fusion copy drift vs golden_v1:\n" + "\n".join(problems)


def test_golden_self_consistency(tmp_path, monkeypatch):
    first = harness.run_chain(harness.LEGACY, tmp_path / "first", monkeypatch)
    second = harness.run_chain(harness.LEGACY, tmp_path / "second", monkeypatch)
    assert _canonical(first) == _canonical(second), (
        "golden capture is not deterministic across two runs"
    )
