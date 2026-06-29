from typer.testing import CliRunner

from market_slice_ml.cli.main import app


def test_cli_status_and_probe_are_offline(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    runner = CliRunner()
    assert runner.invoke(app, ["status"]).exit_code == 0
    result = runner.invoke(app, ["probe"])
    assert result.exit_code == 0
    assert "未送出 live request" in result.stdout
