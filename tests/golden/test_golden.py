"""Golden gate: the fusion pipeline reproduces the frozen ``golden_v1.json``.

Originally captured on the pristine legacy module (S0); from S4 the canonical
snapshot is captured on the **new** ``experts_fusion`` module, whose stochastic
ops use per-unit ``seed_for`` generators (§3.3) and therefore reproduce
deterministically. The new module is held to the whole snapshot; the legacy
module is held only to the deterministic entries (``hold_stochastic=False``) --
it still consumes the global RNG, so after the S4 re-base it no longer matches
the stochastic (Fisher) entries by design. A self-consistency test proves the
capture is deterministic (double regeneration is identical).

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
    # Legacy is the pre-S4 oracle: it still consumes the global RNG, so after the
    # seed_for re-base it reproduces only the deterministic entries. Hold it to
    # those (partition + deterministic methods); skip the stochastic ones.
    aggregates = harness.run_chain(harness.LEGACY, tmp_path, monkeypatch)
    golden = _load_golden()
    _require_golden(golden)
    problems = tolerances.diff_aggregates(aggregates, golden, hold_stochastic=False)
    assert not problems, "legacy module drift vs golden_v1:\n" + "\n".join(problems)


def test_golden_new_module(tmp_path, monkeypatch):
    aggregates = harness.run_chain(harness.NEW, tmp_path, monkeypatch)

    if UPDATE:
        snapshot = {"registry": harness.registry_snapshot(), **aggregates}
        with open(GOLDEN_PATH, "w", encoding="utf-8") as fh:
            json.dump(snapshot, fh, indent=2, sort_keys=True, allow_nan=True)
        return

    golden = _load_golden()
    _require_golden(golden)
    problems = tolerances.diff_aggregates(aggregates, golden, hold_stochastic=True)
    assert not problems, "experts_fusion copy drift vs golden_v1:\n" + "\n".join(problems)


def test_golden_self_consistency(tmp_path, monkeypatch):
    # The new module is the capture source; with seed_for its stochastic ops are
    # deterministic, so two runs must be byte-identical.
    first = harness.run_chain(harness.NEW, tmp_path / "first", monkeypatch)
    second = harness.run_chain(harness.NEW, tmp_path / "second", monkeypatch)
    assert _canonical(first) == _canonical(second), (
        "golden capture is not deterministic across two runs"
    )
