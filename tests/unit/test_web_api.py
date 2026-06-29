from __future__ import annotations

from time import monotonic, sleep

from fastapi.testclient import TestClient

from market_slice_ml.config.settings import Settings
from market_slice_ml.web.app import create_app


def settings(tmp_path, **overrides):
    values = {
        "data_root": tmp_path,
        "no_network": True,
        "web_host": "127.0.0.1",
        "web_api_token": "",
    }
    values.update(overrides)
    return Settings(**values)


def test_web_ui_uses_taiwan_traditional_chinese_and_local_assets(tmp_path):
    with TestClient(create_app(settings(tmp_path))) as client:
        response = client.get("/")
        css = client.get("/static/app.css")
    assert response.status_code == 200
    assert 'lang="zh-Hant-TW"' in response.text
    assert "研究工作區" in response.text
    assert "Symbol" in response.text
    assert "https://" not in response.text + css.text


def test_unconfigured_providers_are_not_reported_as_ready(tmp_path):
    with TestClient(create_app(settings(tmp_path))) as client:
        providers = client.get("/api/status").json()["providers"]
    by_id = {item["provider_id"]: item for item in providers}
    assert by_id["alpha_vantage"]["status"] == "disabled"
    assert by_id["ibkr_realtime"]["status"] == "disabled"


def test_non_local_binding_requires_token(tmp_path):
    try:
        create_app(settings(tmp_path, web_host="0.0.0.0"))
    except RuntimeError as exc:
        assert "WEB_API_TOKEN" in str(exc)
    else:
        raise AssertionError("non-local binding must reject missing token")


def test_api_token_and_validation_errors_are_structured(tmp_path):
    app = create_app(settings(tmp_path, web_api_token="test-token"))
    with TestClient(app) as client:
        unauthorized = client.get("/api/status")
        invalid = client.post(
            "/api/jobs/fetch",
            headers={"X-API-Token": "test-token"},
            json={
                "symbols": ["NVDA"],
                "provider": "yfinance",
                "interval": "5m",
                "start_utc": "2024-01-03T00:00:00Z",
                "end_utc": "2024-01-02T00:00:00Z",
            },
        )
    assert unauthorized.status_code == 401
    assert unauthorized.json()["code"] == "invalid_api_token"
    assert invalid.status_code == 422
    assert invalid.json()["code"] == "request_validation_error"
    assert "操作尚未送出" in invalid.json()["message"]


def test_offline_fetch_job_persists_actionable_failure(tmp_path):
    with TestClient(create_app(settings(tmp_path))) as client:
        submitted = client.post(
            "/api/jobs/fetch",
            json={
                "symbols": ["nvda"],
                "provider": "yfinance",
                "interval": "5m",
                "start_utc": "2024-01-02T00:00:00Z",
                "end_utc": "2024-01-03T00:00:00Z",
            },
        )
        assert submitted.status_code == 202
        job_id = submitted.json()["job_id"]
        deadline = monotonic() + 3.0
        while monotonic() < deadline:
            job = client.get(f"/api/jobs/{job_id}").json()
            if job["status"] not in {"queued", "running"}:
                break
            sleep(0.01)
    assert job["status"] == "failed"
    assert "離線模式" in job["message"]
    assert "NO_NETWORK" in job["suggested_action"]
    assert job["parameters"]["symbols"] == ["NVDA"]


def test_prediction_error_explains_required_state(tmp_path):
    with TestClient(create_app(settings(tmp_path))) as client:
        response = client.post("/api/predictions", json={"symbol": "nq=f", "horizon": "h1"})
    assert response.status_code == 409
    payload = response.json()
    assert payload["code"] == "prediction_unavailable"
    assert "Features" in payload["suggested_action"]
