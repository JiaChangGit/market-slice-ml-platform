from market_slice_ml.config.loader import flatten_symbol_config, load_yaml


def test_settings_yaml_loader_and_symbol_flattening(tmp_path):
    path = tmp_path / "symbols.yaml"
    path.write_text("target_symbols: {group: [AAA, BBB]}\ncontext_symbols: [CCC]\n")
    targets, context = flatten_symbol_config(load_yaml(path))
    assert targets == ["AAA", "BBB"]
    assert context == ["CCC"]
