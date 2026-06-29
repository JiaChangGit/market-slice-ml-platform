import yaml


def test_docker_compose_uses_named_local_data_volume():
    config = yaml.safe_load(open("docker-compose.yml", encoding="utf-8"))
    assert config["services"]["app"]["environment"]["DATA_ROOT"] == "/app/data"
    assert config["services"]["app"]["environment"]["NO_NETWORK"] == "1"
    assert config["volumes"]["market_data"]["name"] == "market_slice_data"
