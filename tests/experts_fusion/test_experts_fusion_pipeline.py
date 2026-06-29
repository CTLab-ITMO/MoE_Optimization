"""Per-stage smoke/contract tests for the collapsed 4-stage experts_fusion module.

Each stage is run in isolation: the upstream artifacts are seeded with the
``common.testing`` builders (so a stage's inputs exist without running the
expensive upstream stage), then the stage is executed from the repo root via the
``run_script`` fixture and its outputs are asserted.
"""
import os

import pytest

from common import layout


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
    builders.experts()
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


def test_stage4_evaluate_accuracy(run_script, builders, data_dir):
    builders.experts()
    pairs = builders.partition()
    builders.fused_experts_v1("Average", pairs)

    run_script("4_evaluate.py", with_model=True, patch_qa_eval=True)

    # Accuracy lands under DATA_DIR (§1.4), not a CWD-relative tree.
    import glob

    matches = glob.glob(
        os.path.join(data_dir, "test_result_*", "accuracy", "accuracy_list_single_Average.pt")
    )
    assert matches, "stage 4 did not write accuracy_list_single_Average under DATA_DIR"


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


def test_stage4_all_replacement_modes(run_script, builders):
    builders.experts()
    pairs = builders.partition()
    builders.fused_experts_v1("Average", pairs)

    run_script(
        "4_evaluate.py",
        with_model=True,
        patch_qa_eval=True,
        config_overrides={
            "REPLACE_LAYERS_SEQUANTIAL": True,
            "REPLACE_BEST_LAYERS_SEQUANTIAL": True,
            "REPLACE_LAYERS_SMART_SEQUANTIAL": True,
        },
    )
