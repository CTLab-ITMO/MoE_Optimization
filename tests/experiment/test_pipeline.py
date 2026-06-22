import os

import pytest


def _lenient(run, name, **kwargs):
    try:
        run(name, **kwargs)
    except Exception as exc:
        message = repr(exc)
        assert "No module named 'config'" not in message, message
        assert "No module named 'common'" not in message, message


def test_stage2_find_best_matches(run_script, builders, data_dir, layers):
    builders.activation_matrices()
    run_script("2_find_best_matches.py")
    for layer in layers:
        assert os.path.exists(os.path.join(data_dir, "pairs", f"{layer}.pt"))


def test_stage6_fusion(run_script, builders, data_dir, layers):
    builders.experts()
    builders.mlp_inputs()
    pairs = builders.pairs()
    builders.pairs_activations(pairs)

    run_script("6_fusion.py")

    for layer in layers:
        assert os.path.exists(
            os.path.join(data_dir, "fusion_experts", "Average", str(layer), "0.pt")
        )


def test_stage1_generate_activation_matrix(run_script, data_dir, layers):
    run_script("1_generate_activation_matrix.py", with_model=True)
    for layer in layers:
        assert os.path.exists(
            os.path.join(data_dir, "activations", f"train_matrix_{layer}.pt")
        )
        assert os.path.exists(
            os.path.join(data_dir, "activations", f"validation_matrix_{layer}.pt")
        )


def test_stage3_save_experts(run_script, data_dir, layers):
    run_script("3_save_experts.py", with_model=True)
    for layer in layers:
        layer_dir = os.path.join(data_dir, "experts", str(layer))
        assert os.path.exists(os.path.join(layer_dir, "expert_0.pt"))
        assert os.path.exists(os.path.join(layer_dir, "gate.pt"))
        assert os.path.exists(os.path.join(layer_dir, "shared_expert.pt"))


def test_stage4_write_mlp_inputs(run_script, data_dir, layers):
    run_script("4_write_mlp_inputs.py", with_model=True)
    for layer in layers:
        assert os.path.exists(os.path.join(data_dir, "mlp_inputs", f"train_{layer}.pt"))
        assert os.path.exists(os.path.join(data_dir, "mlp_inputs", f"validation_{layer}.pt"))


def test_stage5_activations(run_script, builders, data_dir, layers):
    builders.activation_matrices()  
    builders.mlp_inputs()
    builders.pairs()

    run_script("5_activations.py", with_model=True)

    for layer in layers:
        assert os.path.exists(
            os.path.join(data_dir, "activations", f"pairs_train_{layer}.pt")
        )


def test_stage9_accuracy_benchmark(run_script, builders):
    builders.experts()
    pairs = builders.pairs()
    builders.fusion_experts("Average", pairs)

    run_script(
        "9_test.py",
        with_model=True,
        patch_fusion_layer=True,
        patch_qa_eval=True,
    )


def test_stage9_all_replacement_modes(run_script, builders):
    builders.experts()
    pairs = builders.pairs()
    builders.fusion_experts("Average", pairs)

    run_script(
        "9_test.py",
        with_model=True,
        patch_fusion_layer=True,
        patch_qa_eval=True,
        config_overrides={
            "REPLACE_LAYERS_SEQUANTIAL": True,
            "REPLACE_BEST_LAYERS_SEQUANTIAL": True,
            "REPLACE_LAYERS_SMART_SEQUANTIAL": True,
        },
    )


def test_stage7_final_benchmark_smoke(run_script, builders):
    builders.experts()
    builders.mlp_inputs()
    pairs = builders.pairs()
    builders.fusion_experts("ZipIt", pairs)
    _lenient(run_script, "7_final_benchmark.py", patch_fusion_layer=True)


def test_stage8_generation_smoke(run_script):
    _lenient(run_script, "8_generation.py", with_model=True, patch_fusion_layer=True)
