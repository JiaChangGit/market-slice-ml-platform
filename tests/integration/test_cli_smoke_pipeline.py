from typer.testing import CliRunner

from market_slice_ml.cli.main import app


def test_cli_offline_smoke_outputs_json(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    output = tmp_path / "reports" / "cli.html"
    result = CliRunner().invoke(app, ["report", "--out", str(output)])
    assert result.exit_code == 0
    assert output.exists()
    assert "Report 已建立" in result.stdout
