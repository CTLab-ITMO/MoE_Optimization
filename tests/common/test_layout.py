"""Contract test: ``common.layout`` helpers reproduce the frozen path literals.

Before S1 the artifact paths were string literals duplicated across the writing
stage and every reader. This test pins those exact literals (the map captured
from the pre-S1 pipeline) and asserts each layout helper still produces them, so
routing the stages and ``common`` readers through ``layout`` is provably a
behaviour-neutral move -- a typo in a template would surface here, not as a
silent ``FileNotFound`` deep in a run.
"""

from common import layout

DATA = "DATA"


def test_subfolder_constants_are_frozen():
    assert layout.EXPERTS_DIRNAME == "experts"
    assert layout.PAIRS_DIRNAME == "pairs"
    assert layout.MLP_INPUTS_DIRNAME == "mlp_inputs"
    assert layout.ACTIVATIONS_DIRNAME == "activations"
    assert layout.FUSION_EXPERTS_DIRNAME == "fusion_experts"
    assert layout.GATE_FILENAME == "gate.pt"
    assert layout.SHARED_EXPERT_FILENAME == "shared_expert.pt"
    assert layout.SHARED_EXPERT_GATE_FILENAME == "shared_expert_gate.pt"
    assert layout.TRAIN_SPLIT == "train"
    assert layout.VALIDATION_SPLIT == "validation"


def test_expert_layout_matches_frozen_literals():
    assert layout.experts_dir(DATA, 5) == "DATA/experts/5"
    assert layout.expert_path(DATA, 5, 3) == "DATA/experts/5/expert_3.pt"
    # The glob shares the per-expert prefix so the count cannot drift from loads.
    assert layout.expert_glob(DATA, 5) == "DATA/experts/5/expert_*.pt"
    assert layout.gate_path(DATA, 5) == "DATA/experts/5/gate.pt"
    assert layout.shared_expert_path(DATA, 5) == "DATA/experts/5/shared_expert.pt"
    assert layout.shared_expert_gate_path(DATA, 5) == "DATA/experts/5/shared_expert_gate.pt"


def test_pairs_layout_matches_frozen_literals():
    assert layout.pairs_dir(DATA) == "DATA/pairs"
    assert layout.pairs_path(DATA, 7) == "DATA/pairs/7.pt"


def test_mlp_inputs_layout_matches_frozen_literals():
    assert layout.mlp_inputs_dir(DATA) == "DATA/mlp_inputs"
    assert layout.mlp_inputs_path(DATA, "train", 2) == "DATA/mlp_inputs/train_2.pt"
    assert layout.mlp_inputs_path(DATA, "validation", 2) == "DATA/mlp_inputs/validation_2.pt"


def test_activations_layout_matches_frozen_literals():
    assert layout.activations_dir(DATA) == "DATA/activations"
    assert layout.activation_matrix_path(DATA, "train", 1) == "DATA/activations/train_matrix_1.pt"
    assert layout.activation_matrix_path(DATA, "validation", 1) == "DATA/activations/validation_matrix_1.pt"
    assert layout.activations_all_path(DATA, "train", 1) == "DATA/activations/train_all_1.pt"
    assert layout.activations_all_path(DATA, "validation", 1) == "DATA/activations/validation_all_1.pt"
    assert layout.pair_activations_path(DATA, "train", 1) == "DATA/activations/pairs_train_1.pt"
    assert layout.pair_activations_path(DATA, "validation", 1) == "DATA/activations/pairs_validation_1.pt"


def test_fusion_experts_layout_matches_frozen_literals():
    assert layout.fusion_method_dir("ZipIt") == "fusion_experts/ZipIt"
    assert layout.fusion_experts_dir(DATA, "ZipIt", 4) == "DATA/fusion_experts/ZipIt/4"
    assert layout.fusion_expert_path(DATA, "ZipIt", 4, 2) == "DATA/fusion_experts/ZipIt/4/2.pt"


def test_fusion_method_dir_feeds_get_fusion_layer_join():
    # get_fusion_layer_by_index joins {data_dir}/{experts_dir}/{layer}/{i}.pt;
    # fusion_method_dir must make that equal fusion_expert_path.
    method, layer, i = "Average", 3, 1
    joined = f"{DATA}/{layout.fusion_method_dir(method)}/{layer}/{i}.pt"
    assert joined == layout.fusion_expert_path(DATA, method, layer, i)
