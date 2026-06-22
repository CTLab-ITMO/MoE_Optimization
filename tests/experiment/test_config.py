import torch


def test_config_values_are_sane():
    import config

    assert isinstance(config.EXPERTS, int) and config.EXPERTS > 0
    assert isinstance(config.LAYERS, list) and len(config.LAYERS) > 0
    assert all(isinstance(layer, int) for layer in config.LAYERS)
    assert isinstance(config.TOP_K, int) and config.TOP_K > 0
    assert isinstance(config.SEED, int)
    assert isinstance(config.DATA_DIR, str) and config.DATA_DIR
    assert isinstance(config.MODEL_NAME, str)
    assert isinstance(config.device, torch.device)


def test_dataset_specs_have_required_keys():
    import config

    for spec in (config.TRAIN_DATASET_NAME, config.TEST_DATASET_NAME):
        assert "path" in spec
        assert "split" in spec


def test_filter_settings_are_lists_or_none():
    import config

    assert config.FILTER_FUSIONS is None or isinstance(config.FILTER_FUSIONS, list)
    assert config.FILTER_METRICS is None or isinstance(config.FILTER_METRICS, list)


def test_pairing_method_is_supported():
    import config

    assert config.PAIRING_METHOD in [
        "activation_correlation", "random", "output_similarity",
    ]
