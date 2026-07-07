"""Seam-contract, parity and idempotency tests for the collapsed pipeline (S8).

The golden gate proves the new 4-stage module reproduces the legacy 9-stage
module end-to-end (aggregates). These tests pin the *seams* of the collapse at
the raw-artifact level, plus the new stage properties:

* **capture seam** -- the single-pass hooks of stage 1 produce the same
  co-activation matrices, MLP-block inputs and top-k selections as the old
  separate stages 1 / 4 / 5;
* **join seam** -- the per-group ``GroupActivationIndex`` of stage 2 matches the
  old per-pair ``pairs_activations`` (stage 5, second half);
* **load-once** -- replacing then restoring an mlp is lossless, and a load-once
  sweep matches a load-per-method sweep;
* **idempotency / resume** -- a re-run of stage 3 skips fresh artifacts and only
  recomputes a deleted one;
* **config snapshot** -- a later stage fails fast on config drift (§3.6).

Decision #6 / STEPS 6.1: the legacy module was taken off CI (kept on disk, not
deleted), so the capture/join seam tests that re-run the legacy stages to
compare raw artifacts (``*_matches_legacy*``) are skipped via ``LEGACY_OFF_CI``.
The remaining seam tests run only the new module and still gate it.
"""
import os
import runpy
import sys
from pathlib import Path

import pytest
import torch

from common import layout, provenance
from common.testing import (
    N_EXPERTS,
    TOP_K,
    FakeQADataset,
    make_fake_datasets,
    make_fake_transformers,
)
from common.testing.fakes import N_LAYERS

ROOT = Path(__file__).resolve().parents[2]
NEW_DIR = ROOT / "experts_fusion"
LEGACY_DIR = ROOT / "Qwen_MoE_experts_fusion_experiment"

# Decision #6 / STEPS 6.1: the legacy module was removed from CI (kept on disk,
# NOT deleted). The seam tests below that re-run the legacy stages to compare
# raw artifacts new-vs-legacy are skipped; every other seam test here runs only
# the new module and still gates it. Drop this marker to re-run the seam parity
# checks against legacy locally.
LEGACY_OFF_CI = pytest.mark.skip(
    reason="legacy module removed from CI (decision #6 / STEPS 6.1); kept on "
    "disk, new-module seam tests unaffected"
)

SEED = 1234
GOLDEN_LAYERS = list(range(N_LAYERS))

BASE_OVERRIDES = {
    "LAYERS": GOLDEN_LAYERS,
    "EXPERTS": N_EXPERTS,
    "TOP_K": TOP_K,
    "device": torch.device("cpu"),
    "SEED": SEED,
    "EVOLUTION_STEPS": 0,
    "PAIRING_METHOD": "activation_correlation",
    "TRAIN_SIZE": 8,
    "VALIDATION_SIZE": 8,
    "TEST_SIZE": 8,
    "FILTER_FUSIONS": ["Average"],
    "FILTER_METRICS": [],
    "INSPECT_FUSION_METHOD": "Average",
    "USE_DEFAULT_EXPERTS": False,
    "REPLACE_LAYERS_SEQUENTIAL": False,
    "REPLACE_BEST_LAYERS_SEQUENTIAL": False,
    "REPLACE_LAYERS_SMART_SEQUENTIAL": False,
}


def _make_runner(monkeypatch):
    """Return a ``run(stage_dir, config_module, data_dir, workdir, ...)`` callable.

    Sets up the FakeMoE transformers/datasets/qa_eval bridges once and runs an
    individual stage script from a module directory with the golden-like config
    overrides applied to that module's config.
    """
    import importlib

    import seaborn

    monkeypatch.setattr(seaborn, "kdeplot", lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "transformers", make_fake_transformers())
    monkeypatch.setitem(sys.modules, "datasets", make_fake_datasets())
    from common import qa_eval

    monkeypatch.setattr(qa_eval, "load_dataset", lambda *a, **k: FakeQADataset())

    def run(stage_dir, config_module, data_dir, workdir, stage, extra_overrides=None):
        cfg = importlib.import_module(config_module)
        overrides = {**BASE_OVERRIDES, "DATA_DIR": str(data_dir), **(extra_overrides or {})}
        for key, value in overrides.items():
            monkeypatch.setattr(cfg, key, value, raising=False)
        qa_eval._DATA_CACHE.clear()
        workdir.mkdir(parents=True, exist_ok=True)
        monkeypatch.chdir(workdir)
        runpy.run_path(str(Path(stage_dir) / stage), run_name="__main__")

    return run


def _run_new_capture_and_group(run, data_dir, workdir):
    run(NEW_DIR, "experts_fusion.config", data_dir, workdir, "1_preprocess.py")
    run(NEW_DIR, "experts_fusion.config", data_dir, workdir, "2_grouping.py")


def _run_legacy_capture_and_group(run, data_dir, workdir):
    for stage in (
        "1_generate_activation_matrix.py",
        "2_find_best_matches.py",
        "3_save_experts.py",
        "4_write_mlp_inputs.py",
        "5_activations.py",
    ):
        run(LEGACY_DIR, "config", data_dir, workdir, stage)


# --------------------------------------------------------------------------- #
# Capture seam: stage-1 hooks == old stages 1 / 4 / 5.
# --------------------------------------------------------------------------- #
@LEGACY_OFF_CI
def test_capture_coactivation_matches_legacy(monkeypatch, tmp_path):
    from common.artifacts import CoActivationMatrix

    run = _make_runner(monkeypatch)
    data_new, data_legacy = tmp_path / "new", tmp_path / "legacy"
    _run_new_capture_and_group(run, data_new, tmp_path / "wn")
    _run_legacy_capture_and_group(run, data_legacy, tmp_path / "wl")

    for layer in GOLDEN_LAYERS:
        for split in (layout.TRAIN_SPLIT, layout.VALIDATION_SPLIT):
            new_matrix = CoActivationMatrix.load(
                layout.activation_matrix_path(str(data_new), split, layer)
            ).matrix
            legacy_matrix = torch.load(
                layout.activation_matrix_path(str(data_legacy), split, layer)
            )
            assert torch.equal(new_matrix, legacy_matrix), (
                f"co-activation seam drift at layer {layer} split {split}"
            )


@LEGACY_OFF_CI
def test_capture_mlp_inputs_match_legacy(monkeypatch, tmp_path):
    run = _make_runner(monkeypatch)
    data_new, data_legacy = tmp_path / "new", tmp_path / "legacy"
    _run_new_capture_and_group(run, data_new, tmp_path / "wn")
    _run_legacy_capture_and_group(run, data_legacy, tmp_path / "wl")

    for layer in GOLDEN_LAYERS:
        for split in (layout.TRAIN_SPLIT, layout.VALIDATION_SPLIT):
            new_inputs = torch.stack(
                torch.load(layout.mlp_inputs_path(str(data_new), split, layer))
            )
            legacy_inputs = torch.stack(
                torch.load(layout.mlp_inputs_path(str(data_legacy), split, layer))
            )
            assert torch.equal(new_inputs, legacy_inputs), (
                f"mlp_inputs seam drift at layer {layer} split {split}"
            )


@LEGACY_OFF_CI
def test_capture_topk_matches_legacy_stage5(monkeypatch, tmp_path):
    run = _make_runner(monkeypatch)
    data_new, data_legacy = tmp_path / "new", tmp_path / "legacy"
    _run_new_capture_and_group(run, data_new, tmp_path / "wn")
    _run_legacy_capture_and_group(run, data_legacy, tmp_path / "wl")

    for layer in GOLDEN_LAYERS:
        for split in (layout.TRAIN_SPLIT, layout.VALIDATION_SPLIT):
            new_topk = torch.load(layout.activations_all_path(str(data_new), split, layer))
            legacy_topk = torch.load(layout.activations_all_path(str(data_legacy), split, layer))
            assert len(new_topk) == len(legacy_topk)
            for a, b in zip(new_topk, legacy_topk):
                assert torch.equal(a, b), f"top-k seam drift at layer {layer} split {split}"


# --------------------------------------------------------------------------- #
# Join seam: GroupActivationIndex == old pairs_activations (as sets).
# --------------------------------------------------------------------------- #
@LEGACY_OFF_CI
def test_join_matches_legacy_pairs_activations(monkeypatch, tmp_path):
    from common.artifacts import GroupActivationIndex

    run = _make_runner(monkeypatch)
    data_new, data_legacy = tmp_path / "new", tmp_path / "legacy"
    _run_new_capture_and_group(run, data_new, tmp_path / "wn")
    _run_legacy_capture_and_group(run, data_legacy, tmp_path / "wl")

    for layer in GOLDEN_LAYERS:
        for split in (layout.TRAIN_SPLIT, layout.VALIDATION_SPLIT):
            new_index = GroupActivationIndex.load(
                layout.group_activations_path(str(data_new), split, layer)
            ).to_lists()
            legacy_pairs = torch.load(
                layout.pair_activations_path(str(data_legacy), split, layer)
            )
            assert len(new_index) == len(legacy_pairs)
            for group, pair in zip(new_index, legacy_pairs):
                assert sorted(group) == sorted(pair), (
                    f"join seam drift at layer {layer} split {split}"
                )


# --------------------------------------------------------------------------- #
# Config snapshot drift (§3.6).
# --------------------------------------------------------------------------- #
def test_config_snapshot_mismatch_is_caught(monkeypatch, tmp_path):
    run = _make_runner(monkeypatch)
    data_dir, workdir = tmp_path / "data", tmp_path / "work"
    run(NEW_DIR, "experts_fusion.config", data_dir, workdir, "1_preprocess.py")
    # Stage 2 with a drifted SEED must fail fast against the stage-1 snapshot.
    with pytest.raises(provenance.ConfigMismatchError):
        run(
            NEW_DIR,
            "experts_fusion.config",
            data_dir,
            workdir,
            "2_grouping.py",
            extra_overrides={"SEED": SEED + 1},
        )


# --------------------------------------------------------------------------- #
# Idempotency / resume (§3.4).
# --------------------------------------------------------------------------- #
def _fused_paths(data_dir):
    return [
        layout.fusion_expert_path(str(data_dir), "Average", layer, group)
        for layer in GOLDEN_LAYERS
        for group in range(N_EXPERTS // 2)
    ]


def test_stage3_resume_skips_fresh_and_recomputes_deleted(monkeypatch, tmp_path):
    run = _make_runner(monkeypatch)
    data_dir, workdir = tmp_path / "data", tmp_path / "work"
    run(NEW_DIR, "experts_fusion.config", data_dir, workdir, "1_preprocess.py")
    run(NEW_DIR, "experts_fusion.config", data_dir, workdir, "2_grouping.py")
    run(NEW_DIR, "experts_fusion.config", data_dir, workdir, "3_fusion.py")

    paths = _fused_paths(data_dir)
    assert all(os.path.exists(p) for p in paths)
    before = {p: os.path.getmtime(p) for p in paths}

    # Re-run: every group is fresh, so nothing is rewritten.
    run(NEW_DIR, "experts_fusion.config", data_dir, workdir, "3_fusion.py")
    after = {p: os.path.getmtime(p) for p in paths}
    assert after == before, "stage 3 rewrote fresh artifacts instead of skipping"

    # Delete one artifact; the resume recomputes only it.
    victim = paths[0]
    os.remove(victim)
    run(NEW_DIR, "experts_fusion.config", data_dir, workdir, "3_fusion.py")
    assert os.path.exists(victim), "stage 3 did not recompute the deleted artifact"
    for p in paths[1:]:
        assert os.path.getmtime(p) == before[p], "stage 3 rewrote an untouched artifact"


# --------------------------------------------------------------------------- #
# Load-once: full restore is lossless, and load-once == load-per-method.
# --------------------------------------------------------------------------- #
def _build_model_and_artifacts(monkeypatch, tmp_path):
    """Run stages 1-3 once and return (run, data_dir, workdir)."""
    run = _make_runner(monkeypatch)
    data_dir, workdir = tmp_path / "data", tmp_path / "work"
    run(NEW_DIR, "experts_fusion.config", data_dir, workdir, "1_preprocess.py")
    run(NEW_DIR, "experts_fusion.config", data_dir, workdir, "2_grouping.py")
    run(NEW_DIR, "experts_fusion.config", data_dir, workdir, "3_fusion.py")
    return run, data_dir, workdir


def test_load_once_full_restore_is_lossless(monkeypatch, tmp_path):
    from common.fusion_layer import get_fusion_layer_from_groups
    from common.testing.fakes import FakeFusionGate

    _, data_dir, _ = _build_model_and_artifacts(monkeypatch, tmp_path)
    # The eval gate is a FakeFusionGate (the harness substitutes it); do the same
    # so the fusion layer can decode the 3-tuple gate output.
    torch.manual_seed(SEED)
    for layer in GOLDEN_LAYERS:
        torch.save(
            FakeFusionGate(), os.path.join(str(data_dir), "experts", str(layer), "gate.pt")
        )

    from common.testing import make_fake_transformers

    model = make_fake_transformers().AutoModelForCausalLM.from_pretrained("x")
    layer = GOLDEN_LAYERS[0]
    original = model.model.layers[layer].mlp
    x = torch.randn(1, 4, 4)
    before = original(x.clone())

    fused = get_fusion_layer_from_groups(
        layer, "Average", str(data_dir), device=torch.device("cpu"),
        use_default_experts=False, num_experts=N_EXPERTS, top_k=TOP_K,
    )
    model.model.layers[layer].mlp = fused
    model.model.layers[layer].mlp = original  # restore
    after = model.model.layers[layer].mlp(x.clone())
    assert torch.equal(before, after), "restoring the original mlp was not lossless"


def test_load_once_matches_load_per_method(monkeypatch, tmp_path):
    from common.fusion_layer import get_fusion_layer_from_groups
    from common.qa_eval import evaluate_qa
    from common.testing import make_fake_transformers
    from common.testing.fakes import FakeFusionGate, FakeTokenizer

    run, data_dir, _ = _build_model_and_artifacts(monkeypatch, tmp_path)
    torch.manual_seed(SEED)
    for layer in GOLDEN_LAYERS:
        torch.save(
            FakeFusionGate(), os.path.join(str(data_dir), "experts", str(layer), "gate.pt")
        )

    tokenizer = FakeTokenizer()
    dataset = {"path": "cais/mmlu", "name": "all", "split": "test"}

    def build(model, layer):
        return get_fusion_layer_from_groups(
            layer, "Average", str(data_dir), device=torch.device("cpu"),
            use_default_experts=False, num_experts=N_EXPERTS, top_k=TOP_K,
        )

    def fresh_model():
        # Seed each construction so the load-per-method models are byte-identical
        # to the load-once model: the property under test is "swap+restore on one
        # model" == "a fresh model per method" on the SAME base weights. Now that
        # logits are a deterministic function of the routed path, drifting base
        # weights would diverge here -- previously masked because logits were
        # RNG-reseeded and every accuracy was the same constant.
        torch.manual_seed(SEED)
        return make_fake_transformers().AutoModelForCausalLM.from_pretrained("x")

    # load-per-method: a fresh model per single-layer replacement.
    per_method = []
    for layer in GOLDEN_LAYERS:
        model = fresh_model()
        model.model.layers[layer].mlp = build(model, layer)
        per_method.append(evaluate_qa(model, tokenizer, dataset, 8, seed=SEED))

    # load-once: one model, replace + restore per layer.
    model = fresh_model()
    originals = {l: model.model.layers[l].mlp for l in GOLDEN_LAYERS}
    once = []
    for layer in GOLDEN_LAYERS:
        model.model.layers[layer].mlp = build(model, layer)
        once.append(evaluate_qa(model, tokenizer, dataset, 8, seed=SEED))
        model.model.layers[layer].mlp = originals[layer]

    assert once == per_method, "load-once accuracy diverged from load-per-method"


def test_fake_model_logits_depend_on_mlp():
    """FakeCausalLM logits are a deterministic function of the routed path.

    The load-once accuracy gate above is only content-ful if swapping an mlp
    actually moves the logits -- otherwise ``once == per_method`` would hold
    vacuously, as it did when the fake returned RNG-reseeded logits independent
    of the model. Pin both properties directly: (1) determinism -- the same input
    twice gives identical logits; (2) sensitivity -- replacing one layer's mlp
    changes the logits, so accuracy cannot be constant across fusion methods.
    """
    from common.testing import make_fake_transformers

    torch.manual_seed(SEED)
    model = make_fake_transformers().AutoModelForCausalLM.from_pretrained("x")
    ids = torch.arange(1, 5).unsqueeze(0)

    logits = model(input_ids=ids).logits
    assert torch.equal(logits, model(input_ids=ids).logits), "fake logits are not deterministic"

    class _Zero(torch.nn.Module):
        def forward(self, x):
            return torch.zeros_like(x)

    model.model.layers[0].mlp = _Zero()
    assert not torch.equal(logits, model(input_ids=ids).logits), (
        "replacing a layer's mlp did not change the logits -- the accuracy gate "
        "would be a tautology"
    )


# --------------------------------------------------------------------------- #
# Full chain with a Best_<metric> pseudo-method (not covered by the golden).
# --------------------------------------------------------------------------- #
def test_full_chain_produces_best_metric_artifacts(monkeypatch, tmp_path):
    run = _make_runner(monkeypatch)
    data_dir, workdir = tmp_path / "data", tmp_path / "work"
    overrides = {"FILTER_METRICS": ["MSE"]}
    for stage in ("1_preprocess.py", "2_grouping.py", "3_fusion.py"):
        run(NEW_DIR, "experts_fusion.config", data_dir, workdir, stage, overrides)
    # Stage 3 wrote a Best_MSE fused expert for every group.
    for layer in GOLDEN_LAYERS:
        for group in range(N_EXPERTS // 2):
            assert os.path.exists(
                layout.fusion_expert_path(str(data_dir), "Best_MSE", layer, group)
            )
    # Stage 4 evaluates it and writes the accuracy list under DATA_DIR.
    run(NEW_DIR, "experts_fusion.config", data_dir, workdir, "4_evaluate.py", overrides)
    import glob

    assert glob.glob(
        os.path.join(str(data_dir), "test_result_*", "accuracy", "accuracy_list_single_Best_MSE.pt")
    )

