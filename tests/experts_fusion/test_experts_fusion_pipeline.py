"""Per-stage smoke/contract tests for the collapsed 4-stage experts_fusion module.

Each stage is run in isolation: the upstream artifacts are seeded with the
``common.testing`` builders (so a stage's inputs exist without running the
expensive upstream stage), then the stage is executed from the repo root via the
``run_script`` fixture and its outputs are asserted.
"""
import glob
import os

import pytest
import torch

from common import layout
from common.expert_adapter import NoSharedExpertAdapter
from common.names import ReplaceMode, TEST_RESULT_PREFIX, ZERO_FUSION


def test_stage1_preprocess(run_script, data_dir, layers):
    run_script("1_preprocess.py", with_model=True)
    for layer in layers:
        experts_dir = layout.experts_dir(data_dir, layer)
        assert os.path.exists(os.path.join(experts_dir, "expert_0.pt"))
        assert os.path.exists(os.path.join(experts_dir, "gate.pt"))
        assert os.path.exists(os.path.join(experts_dir, "shared_expert.pt"))
        for split in (layout.TRAIN_SPLIT, layout.VALIDATION_SPLIT):
            assert os.path.exists(layout.activation_matrix_path(data_dir, split, layer))
            assert os.path.exists(layout.mlp_inputs_path(data_dir, split, layer))
            assert os.path.exists(layout.activations_all_path(data_dir, split, layer))
    # Provenance (§3.6): config snapshot, run manifest and the layout version.
    assert os.path.exists(os.path.join(data_dir, "config_snapshot.json"))
    assert os.path.exists(os.path.join(data_dir, "run_manifest.json"))
    assert layout.read_layout_version(data_dir) == layout.LAYOUT_VERSION


def test_stage1_no_shared_expert_family_writes_no_shared_files(run_script, data_dir, layers):
    """A family without a shared expert (NoSharedExpertAdapter) writes gate + experts
    but no shared_expert files; stage 4's loader later falls back to a zeros term."""
    run_script(
        "1_preprocess.py",
        with_model=True,
        config_overrides={"SHARED_EXPERT_ADAPTER": NoSharedExpertAdapter()},
    )
    for layer in layers:
        experts_dir = layout.experts_dir(data_dir, layer)
        assert os.path.exists(os.path.join(experts_dir, "expert_0.pt"))
        assert os.path.exists(os.path.join(experts_dir, "gate.pt"))
        assert not os.path.exists(layout.shared_expert_path(data_dir, layer))
        assert not os.path.exists(layout.shared_expert_gate_path(data_dir, layer))


def test_stage1_records_produced_layers(run_script, data_dir, layers):
    """Stage 1 records the MoE layers it captured into the run manifest (CR#4)."""
    run_script("1_preprocess.py", with_model=True)
    from common import provenance

    manifest = provenance.RunManifest.load(data_dir)
    assert manifest is not None
    # On the fake zoo every configured layer is MoE, so produced == LAYERS.
    assert manifest.produced_layers == list(layers)


def test_stage2_iterates_manifest_produced_layers(run_script, builders, data_dir, layers):
    """CR#4: stage 2 groups only the layers stage 1 recorded, not the raw LAYERS.

    The manifest claims only the first layer was produced (as if the rest were
    dense and skipped by discover_moe_layers); the config LAYERS still lists all.
    Stage 2 must group the recorded layer and leave the others untouched.
    """
    from common import provenance

    builders.coactivation_matrices()
    builders.topk_all()
    provenance.RunManifest(seed=1, produced_layers=[layers[0]]).save(data_dir)

    run_script("2_grouping.py")

    assert os.path.exists(layout.group_path(data_dir, layers[0]))
    for skipped in layers[1:]:
        assert not os.path.exists(layout.group_path(data_dir, skipped)), (
            "stage 2 grouped a layer absent from the manifest's produced_layers"
        )


def test_stage3_iterates_manifest_produced_layers(run_script, builders, data_dir, layers):
    """CR#4: stage 3 fuses only the layers stage 1 recorded, not the raw LAYERS."""
    from common import provenance

    builders.experts(saved=True)
    builders.mlp_inputs()
    pairs = builders.partition()
    builders.group_activations(n_groups=len(pairs))
    provenance.RunManifest(seed=1, produced_layers=[layers[0]]).save(data_dir)

    run_script("3_fusion.py")

    assert os.path.exists(layout.fusion_expert_path(data_dir, "Average", layers[0], 0))
    for skipped in layers[1:]:
        assert not os.path.exists(
            layout.fusion_expert_path(data_dir, "Average", skipped, 0)
        ), "stage 3 fused a layer absent from the manifest's produced_layers"


def test_stage1_coactivation_is_versioned(run_script, data_dir, layers):
    run_script("1_preprocess.py", with_model=True)
    from common.artifacts import CoActivationMatrix

    # The matrix is a versioned CoActivationMatrix (loads via the class, not a raw
    # pickle), and carries its split metadata.
    matrix = CoActivationMatrix.load(
        layout.activation_matrix_path(data_dir, layout.TRAIN_SPLIT, layers[0])
    )
    assert matrix.split == layout.TRAIN_SPLIT
    assert matrix.num_experts > 0


def test_stage2_grouping(run_script, builders, data_dir, layers):
    builders.coactivation_matrices()
    builders.topk_all()
    run_script("2_grouping.py")
    from common.artifacts import GroupActivationIndex, Partition

    for layer in layers:
        partition = Partition.load(layout.group_path(data_dir, layer))
        assert partition.num_groups == builders.n_experts // 2
        for split in (layout.TRAIN_SPLIT, layout.VALIDATION_SPLIT):
            index = GroupActivationIndex.load(
                layout.group_activations_path(data_dir, split, layer)
            )
            assert index.num_groups == partition.num_groups


def test_stage3_fusion(run_script, builders, data_dir, layers):
    builders.experts(saved=True)
    builders.mlp_inputs()
    pairs = builders.partition()
    builders.group_activations(n_groups=len(pairs))

    run_script("3_fusion.py")

    from common.fused_group import FusedExpert

    for layer in layers:
        path = layout.fusion_expert_path(data_dir, "Average", layer, 0)
        assert os.path.exists(path)
        # The fused artifact is a versioned FusedExpert (loads via the class).
        assert FusedExpert.load(path).group_size == 2


def test_stage3_and_loader_handle_singleton_and_triple_groups(
    run_script, builders, data_dir, layers
):
    """A partition with a singleton and a triple flows through stage 3 (k-agnostic
    methods) into identity-contract artifacts, and the stage-4 loader serves them
    through a forward-capable FusionLayer."""
    from common.artifacts import Group, Partition
    from common.fused_group import (
        IDENTITY_RECONSTRUCTION,
        FusedExpert,
        IdentityFusedGroup,
    )
    from common.fusion_layer import get_fusion_layer_from_groups
    from common.router import Qwen15RouterAdapter
    from common.testing import N_EXPERTS, TOP_K
    from common.testing.fakes import H

    builders.experts(saved=True)
    builders.mlp_inputs()
    groups = [(0,), (1, 2, 3)]
    for layer in layers:
        path = layout.group_path(data_dir, layer)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        Partition(
            groups=tuple(Group(members=g) for g in groups),
            method="activation_correlation",
        ).save(path)
    builders.group_activations(n_groups=len(groups))

    run_script("3_fusion.py", config_overrides={"FILTER_FUSIONS": ["Average", "Pruning"]})

    for method in ("Average", "Pruning"):
        for i, members in enumerate(groups):
            fused = FusedExpert.load(
                layout.fusion_expert_path(data_dir, method, layers[0], i)
            )
            assert fused.group_size == len(members)
            assert fused.reconstruction == IDENTITY_RECONSTRUCTION

    fusion = get_fusion_layer_from_groups(
        layers[0], "Average", data_dir, device=torch.device("cpu"),
        use_default_experts=False, num_experts=N_EXPERTS,
        router_adapter=Qwen15RouterAdapter(), top_k=TOP_K,
    )
    assert [group.group_size for group in fusion.fusion_groups] == [1, 3]
    assert all(isinstance(group, IdentityFusedGroup) for group in fusion.fusion_groups)
    torch.manual_seed(0)
    out = fusion.forward(torch.randn(1, 5, H))
    assert out.shape == (1, 5, H)
    assert torch.isfinite(out).all()


def test_stage4_evaluate_accuracy(run_script, builders, data_dir):
    builders.experts(saved=True)
    pairs = builders.partition()
    builders.fused_experts_v1("Average", pairs)

    run_script("4_evaluate.py", with_model=True, patch_qa_eval=True)

    # Accuracy lands under DATA_DIR (§1.4), not a CWD-relative tree.
    import glob

    matches = glob.glob(
        os.path.join(data_dir, "test_result_*", "accuracy", "accuracy_list_single_Average.pt")
    )
    assert matches, "stage 4 did not write accuracy_list_single_Average under DATA_DIR"


def test_stage4_evaluate_without_shared_expert(run_script, builders, data_dir, layers):
    """Stage 4 evaluates a family without a shared expert: with the shared files
    removed the loader falls back to a zeros shared term instead of raising."""
    builders.experts(saved=True)
    pairs = builders.partition()
    builders.fused_experts_v1("Average", pairs)
    for layer in layers:
        os.remove(layout.shared_expert_path(data_dir, layer))
        os.remove(layout.shared_expert_gate_path(data_dir, layer))

    run_script(
        "4_evaluate.py",
        with_model=True,
        patch_qa_eval=True,
        config_overrides={"SHARED_EXPERT_ADAPTER": NoSharedExpertAdapter()},
    )

    matches = glob.glob(
        os.path.join(data_dir, "test_result_*", "accuracy", "accuracy_list_single_Average.pt")
    )
    assert matches, "stage 4 did not evaluate a model without a shared expert"


def test_run_stage_files_orders_and_filters():
    from experts_fusion import run

    assert [s.name for s in run.stage_files()] == [
        "1_preprocess.py",
        "2_grouping.py",
        "3_fusion.py",
        "4_evaluate.py",
    ]
    assert [s.name for s in run.stage_files(["1", "3"])] == [
        "1_preprocess.py",
        "3_fusion.py",
    ]


def test_stage4_all_replacement_modes(run_script, builders, data_dir, layers):
    builders.experts(saved=True)
    pairs = builders.partition()
    builders.fused_experts_v1("Average", pairs)

    run_script(
        "4_evaluate.py",
        with_model=True,
        patch_qa_eval=True,
        config_overrides={
            "REPLACE_LAYERS_SEQUENTIAL": True,
            "REPLACE_BEST_LAYERS_SEQUENTIAL": True,
            "REPLACE_LAYERS_SMART_SEQUENTIAL": True,
        },
    )

    # A smoke run that only checks "did not crash" hides broken replacement modes
    # (CR#13). Assert each mode's artifacts exist and each accuracy/size list has
    # the expected length instead. With FILTER_FUSIONS=["Average"] and
    # FILTER_METRICS=[] the evaluated names are exactly Average + the zero baseline
    # (no Best_<metric> pseudo-methods), and every sweep records base + one entry
    # per MoE layer.
    result_dirs = glob.glob(os.path.join(data_dir, f"{TEST_RESULT_PREFIX}*"))
    assert len(result_dirs) == 1, result_dirs
    accuracy_dir = os.path.join(result_dirs[0], "accuracy")
    model_size_dir = os.path.join(result_dirs[0], "model_size")
    expected_len = len(layers) + 1

    # SEQ comes from REPLACE_LAYERS_SEQUENTIAL / REPLACE_BEST_LAYERS_SEQUENTIAL;
    # SMART_SEQ from REPLACE_LAYERS_SMART_SEQUENTIAL. SINGLE must NOT appear because
    # REPLACE_LAYERS_SEQUENTIAL flips the per-layer mode off single.
    for name in ("Average", ZERO_FUSION):
        for mode in (ReplaceMode.SEQ, ReplaceMode.SMART_SEQ):
            accuracy_path = os.path.join(accuracy_dir, f"accuracy_list_{mode}_{name}.pt")
            model_size_path = os.path.join(model_size_dir, f"model_size_list_{mode}_{name}.pt")
            assert os.path.exists(accuracy_path), f"missing {accuracy_path}"
            assert os.path.exists(model_size_path), f"missing {model_size_path}"
            assert len(torch.load(accuracy_path)) == expected_len
            assert len(torch.load(model_size_path)) == expected_len

    assert not glob.glob(os.path.join(accuracy_dir, f"accuracy_list_{ReplaceMode.SINGLE}_*.pt")), (
        "single-mode artifacts must not be produced when REPLACE_LAYERS_SEQUENTIAL is on"
    )


def test_stage4_does_not_load_validation_mlp_inputs(run_script, builders, data_dir, monkeypatch):
    """Regression guard for FR §5-3: the dead validation-mlp_inputs load stays removed.

    Stage 4's reconstruction benchmark reads only the TRAIN split; the legacy
    stage 7 also loaded (and never used) the validation split. Seed both splits so
    an accidental validation load would succeed silently, then spy on torch.load
    and assert the train inputs are read while the validation inputs never are.
    """
    builders.experts(saved=True)
    pairs = builders.partition()
    builders.fused_experts_v1("Average", pairs)
    builders.mlp_inputs()  # writes BOTH train_ and validation_ mlp_inputs

    loaded = []
    real_load = torch.load

    def spy_load(path, *args, **kwargs):
        loaded.append(str(path))
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", spy_load)

    run_script("4_evaluate.py", with_model=True, patch_qa_eval=True)

    inputs_dir = layout.mlp_inputs_dir(data_dir).replace("\\", "/")
    normalized = [p.replace("\\", "/") for p in loaded]
    train_loaded = [p for p in normalized if p.startswith(inputs_dir) and "/train_" in p]
    validation_loaded = [p for p in normalized if p.startswith(inputs_dir) and "/validation_" in p]
    assert train_loaded, "recon benchmark should load train mlp_inputs (else the guard is vacuous)"
    assert not validation_loaded, f"stage 4 must not load validation mlp_inputs, got {validation_loaded}"


def test_stage4_eval_top_k_tracks_config(run_script, builders, monkeypatch):
    """Regression guard for FR §5-4: eval builds fusion layers with config.TOP_K.

    The legacy code defaulted the eval-time top-k to a literal (8) that could drift
    from config.TOP_K. Override TOP_K to a distinctive value and assert every
    fusion layer stage 4 builds carries it, so a re-introduced literal is caught.
    """
    builders.experts(saved=True)
    pairs = builders.partition()
    builders.fused_experts_v1("Average", pairs)

    from common import fusion_layer

    distinctive_top_k = 3  # not equal to any literal the eval path might hardcode
    seen_top_k = []
    real = fusion_layer.get_fusion_layer_from_groups

    def spy(*args, **kwargs):
        seen_top_k.append(kwargs.get("top_k"))
        return real(*args, **kwargs)

    monkeypatch.setattr(fusion_layer, "get_fusion_layer_from_groups", spy)

    run_script(
        "4_evaluate.py",
        with_model=True,
        patch_qa_eval=True,
        config_overrides={"TOP_K": distinctive_top_k},
    )

    assert seen_top_k, "stage 4 never built a fusion layer"
    assert all(top_k == distinctive_top_k for top_k in seen_top_k), (
        f"eval built layers with top_k={set(seen_top_k)}, expected config.TOP_K={distinctive_top_k}"
    )
