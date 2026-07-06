import os
import types

import pytest
import torch


def test_config_reexports_the_selected_default():
    import experts_fusion.config as config
    from experts_fusion.configs import default

    assert config._CONFIG_NAME == "default"
    assert config.DATA_DIR == default.DATA_DIR
    assert config.MODEL_NAME == default.MODEL_NAME
    assert config.config_snapshot_dict()["EXPERTS"] == default.EXPERTS


def test_run_config_flag_sets_env_and_parses_stages(monkeypatch):
    import experts_fusion.run as run

    monkeypatch.delenv("EXPERTS_FUSION_CONFIG", raising=False)
    ran = {}
    monkeypatch.setattr(run, "stage_files", lambda only=None: (ran.setdefault("only", only), [])[1])

    run.main(["--config", "myexp", "1", "2"])

    assert os.environ["EXPERTS_FUSION_CONFIG"] == "myexp"
    assert ran["only"] == ["1", "2"]
    monkeypatch.delenv("EXPERTS_FUSION_CONFIG", raising=False)


def test_config_values_are_sane():
    import experts_fusion.config as config

    assert isinstance(config.EXPERTS, int) and config.EXPERTS > 0
    assert isinstance(config.LAYERS, list) and len(config.LAYERS) > 0
    assert all(isinstance(layer, int) for layer in config.LAYERS)
    assert isinstance(config.TOP_K, int) and config.TOP_K > 0
    assert isinstance(config.SEED, int)
    assert isinstance(config.DATA_DIR, str) and config.DATA_DIR
    assert isinstance(config.MODEL_NAME, str)
    assert isinstance(config.device, torch.device)


def test_dataset_specs_have_required_keys():
    import experts_fusion.config as config

    for spec in (config.TRAIN_DATASET_NAME, config.TEST_DATASET_NAME):
        assert "path" in spec
        assert "split" in spec


def test_filter_settings_are_lists_or_none():
    import experts_fusion.config as config

    assert config.FILTER_FUSIONS is None or isinstance(config.FILTER_FUSIONS, list)
    assert config.FILTER_METRICS is None or isinstance(config.FILTER_METRICS, list)


def test_pairing_method_is_supported():
    import experts_fusion.config as config

    assert config.PAIRING_METHOD in [
        "activation_correlation", "random", "output_similarity",
    ]


def test_data_dir_is_scoped_under_data_root():
    import experts_fusion.config as config

    # New module writes under ./data/ (run from the repo root), not the legacy
    # module-local ./data_MMLU_1k.
    assert config.DATA_DIR.replace("\\", "/").startswith("./data/")


def test_config_snapshot_covers_every_config_field():
    # Every UPPER_CASE config knob must be in the snapshot (so config drift is
    # detected) or be explicitly excluded -- a new field can't silently escape.
    # Imported registries are tools, not config fields, so they don't count.
    import experts_fusion.config as config
    from common.registry import Registry

    fields = {
        name
        for name in dir(config)
        if name.isupper()
        and not name.startswith("_")
        and not isinstance(getattr(config, name), Registry)
    }
    covered = set(config.config_snapshot_dict()) | config._SNAPSHOT_EXCLUDED
    missing = fields - covered
    assert not missing, f"config fields missing from snapshot/excluded: {sorted(missing)}"


# --------------------------------------------------------------------------- #
# Value tests for the already-extracted config knobs (Требования §6): assert
# each knob's *effect* on a stage, not just its type. These were flagged as
# proven-by-absence only (FINAL_REVIEW §5 debt #6); a hardcoded literal replacing
# any of them would slip past the structural tests above but fail one of these.
# --------------------------------------------------------------------------- #
def test_dataset_text_field_is_read_from_config(run_script):
    """Stage 1 must index the dataset by ``config.DATASET_TEXT_FIELD``."""
    accessed_keys = []

    class RecordingSplit(dict):
        def __getitem__(self, key):
            accessed_keys.append(key)
            return super().__getitem__(key)

    payload = RecordingSplit({"custom_field": [f"q{i}" for i in range(8)]})
    datasets_module = types.ModuleType("datasets")
    datasets_module.load_dataset = lambda *a, **k: payload

    run_script(
        "1_preprocess.py",
        with_model=True,
        datasets_module=datasets_module,
        config_overrides={"DATASET_TEXT_FIELD": "custom_field"},
    )

    assert accessed_keys, "stage 1 never indexed the dataset"
    assert all(key == "custom_field" for key in accessed_keys), (
        f"stage 1 indexed the dataset by {set(accessed_keys)}, "
        "expected only config.DATASET_TEXT_FIELD"
    )


def test_min_pair_validation_samples_gates_metric_computation(run_script, builders, monkeypatch):
    """Stage 3 scores a group on validation data only when it has more than
    ``config.MIN_PAIR_VALIDATION_SAMPLES`` activating examples."""
    builders.experts(saved=True)
    builders.mlp_inputs()  # 16 captured inputs per layer
    pairs = builders.partition()
    builders.group_activations(n_groups=len(pairs))  # 16 activating indices per group

    from common import mlp_utils

    calls = []
    real = mlp_utils.test_model_on_data

    def spy_test_model_on_data(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(mlp_utils, "test_model_on_data", spy_test_model_on_data)

    # Threshold above the 16 validation samples per group -> metrics skipped.
    run_script("3_fusion.py", config_overrides={"MIN_PAIR_VALIDATION_SAMPLES": 100})
    assert not calls, "validation metrics computed despite threshold above the sample count"

    # Threshold below the sample count -> metrics computed. The knob is part of the
    # stage-3 idempotency hash, so changing it re-fuses instead of resuming.
    calls.clear()
    run_script("3_fusion.py", config_overrides={"MIN_PAIR_VALIDATION_SAMPLES": 0})
    assert calls, "validation metrics skipped despite threshold below the sample count"


def test_use_default_experts_is_threaded_into_eval_layers(run_script, builders, monkeypatch):
    """Stage 4 builds its evaluation layers with ``config.USE_DEFAULT_EXPERTS``."""
    builders.experts(saved=True)
    pairs = builders.partition()
    builders.fused_experts_v1("Average", pairs)
    # No mlp_inputs seeded: the recon benchmark (which hardcodes
    # use_default_experts=True) is skipped, so every built layer reflects the knob.

    from common import fusion_layer

    real = fusion_layer.get_fusion_layer_from_groups

    def run_with(value):
        seen = []

        def spy(*args, **kwargs):
            seen.append(kwargs.get("use_default_experts"))
            return real(*args, **kwargs)

        monkeypatch.setattr(fusion_layer, "get_fusion_layer_from_groups", spy)
        run_script(
            "4_evaluate.py",
            with_model=True,
            patch_qa_eval=True,
            config_overrides={"USE_DEFAULT_EXPERTS": value},
        )
        return seen

    seen_true = run_with(True)
    assert seen_true and all(value is True for value in seen_true), seen_true
    seen_false = run_with(False)
    assert seen_false and all(value is False for value in seen_false), seen_false


def test_inspect_fusion_method_is_validated_at_stage_start(run_script, builders):
    """Stage 4 rejects an ``config.INSPECT_FUSION_METHOD`` no stage-3 method produced."""
    builders.experts(saved=True)
    pairs = builders.partition()
    builders.fused_experts_v1("Average", pairs)

    with pytest.raises(ValueError, match="INSPECT_FUSION_METHOD"):
        run_script(
            "4_evaluate.py",
            with_model=True,
            patch_qa_eval=True,
            config_overrides={"INSPECT_FUSION_METHOD": "NoSuchMethod"},
        )


def test_model_dtype_is_used_to_load_the_model(run_script, builders):
    """Stage 4 loads the base model with ``config.MODEL_DTYPE`` (not a literal)."""
    from common.testing import FakeCausalLM, FakeTokenizer

    builders.experts(saved=True)
    pairs = builders.partition()
    builders.fused_experts_v1("Average", pairs)

    captured = {}
    transformers_module = types.ModuleType("transformers")

    def _from_pretrained(*a, **k):
        captured["dtype"] = k.get("dtype")
        return FakeCausalLM()

    transformers_module.AutoModelForCausalLM = types.SimpleNamespace(
        from_pretrained=_from_pretrained
    )
    transformers_module.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: FakeTokenizer()
    )

    run_script(
        "4_evaluate.py",
        with_model=True,
        patch_qa_eval=True,
        transformers_module=transformers_module,
        config_overrides={"MODEL_DTYPE": torch.float16},
    )

    assert captured.get("dtype") == torch.float16
