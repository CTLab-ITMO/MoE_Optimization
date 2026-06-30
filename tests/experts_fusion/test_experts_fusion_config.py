import torch


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
