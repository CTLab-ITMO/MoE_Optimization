"""Golden gate: the fusion pipeline reproduces the frozen ``golden_v1.json``.

Originally captured on the pristine legacy module (S0); from S4 the canonical
snapshot is captured on the **new** ``experts_fusion`` module, whose stochastic
ops use per-unit ``seed_for`` generators (§3.3) and therefore reproduce
deterministically. The new module is held to the snapshot except the
platform-sensitive methods (Procrustes/ZipIt -- see tolerances); the legacy
module is held to fewer entries still (it draws from the global RNG, so it no
longer matches the stochastic Fisher entries). The two modules are checked
against each other directly by ``test_golden_legacy_matches_new`` -- the real
equivalence guarantee, which also covers the platform-sensitive methods. A
self-consistency test proves the capture is deterministic (double regeneration
is identical).

Re-basing the golden is intentional and rare: run with ``UPDATE_GOLDEN=1`` and
review the JSON diff in the PR. The plan sanctions exactly two re-bases (S4
local seeds, S5 ``Best_*`` fallback); any other change to values / orientation /
order is a red gate.

Decision #6 / STEPS 6.1: the legacy module was taken off CI (kept on disk, not
deleted), so the two tests that run the legacy chain
(``test_golden_legacy_module``, ``test_golden_legacy_matches_new``) are skipped
via ``LEGACY_OFF_CI``. What still gates the new module is unchanged:
``test_golden_new_module`` holds it to ``golden_v1.json`` and
``test_golden_self_consistency`` proves the capture is deterministic.
"""
import json
import os
from pathlib import Path

import pytest

import harness
import tolerances

GOLDEN_PATH = Path(__file__).resolve().parent / "golden_v1.json"
UPDATE = bool(os.environ.get("UPDATE_GOLDEN"))

# Decision #6 / STEPS 6.1: the legacy ``Qwen_MoE_experts_fusion_experiment``
# module was removed from CI (kept on disk, NOT deleted). Every test that runs
# the legacy chain is skipped; the new module's golden oracle
# (``test_golden_new_module`` against ``golden_v1.json``) is unaffected. Drop
# this marker to re-run the legacy equivalence gate locally.
LEGACY_OFF_CI = pytest.mark.skip(
    reason="legacy module removed from CI (decision #6 / STEPS 6.1); kept on "
    "disk, new-module golden oracle unaffected"
)


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


@LEGACY_OFF_CI
def test_golden_legacy_module(tmp_path, monkeypatch):
    # Legacy is the pre-S4 oracle: it still draws from the global RNG, so it no
    # longer reproduces the new module's seed_for entries -- skip the stochastic
    # ones. Also skip the platform-sensitive methods (Procrustes/ZipIt), whose
    # absolute values are not portable across OS/CPU.
    aggregates = harness.run_chain(harness.LEGACY, tmp_path, monkeypatch)
    golden = _load_golden()
    _require_golden(golden)
    skip = tolerances.STOCHASTIC_FUSIONS | tolerances.PLATFORM_SENSITIVE_FUSIONS
    problems = tolerances.diff_aggregates(aggregates, golden, skip=skip)
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
    # The new module reproduces the stochastic entries (seed_for), so they stay
    # in the snapshot under loose tolerance; only the platform-sensitive methods
    # (Procrustes/ZipIt) are skipped (not portable across OS/CPU).
    problems = tolerances.diff_aggregates(
        aggregates, golden, skip=tolerances.PLATFORM_SENSITIVE_FUSIONS
    )
    assert not problems, "experts_fusion copy drift vs golden_v1:\n" + "\n".join(problems)


@LEGACY_OFF_CI
def test_golden_legacy_matches_new(tmp_path, monkeypatch):
    # Equivalence is the golden's real purpose. Running both modules on the SAME
    # machine, the platform-sensitive methods (Procrustes/ZipIt) agree exactly,
    # so this guards them even though the frozen snapshot cannot. Only stochastic
    # methods are skipped (legacy uses the global RNG, new uses seed_for).
    legacy = harness.run_chain(harness.LEGACY, tmp_path / "legacy", monkeypatch)
    new = harness.run_chain(harness.NEW, tmp_path / "new", monkeypatch)
    problems = tolerances.diff_aggregates(
        legacy, new, skip=tolerances.STOCHASTIC_FUSIONS
    )
    assert not problems, "legacy vs new module drift:\n" + "\n".join(problems)


def test_golden_self_consistency(tmp_path, monkeypatch):
    # The new module is the capture source; with seed_for its stochastic ops are
    # deterministic, so two runs must be byte-identical.
    first = harness.run_chain(harness.NEW, tmp_path / "first", monkeypatch)
    second = harness.run_chain(harness.NEW, tmp_path / "second", monkeypatch)
    assert _canonical(first) == _canonical(second), (
        "golden capture is not deterministic across two runs"
    )
