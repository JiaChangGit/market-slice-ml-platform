# ruff: noqa: E501
"""UTF-8 self-contained research report with no remote assets."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment

from market_slice_ml.domain.models import PredictionRecord

TEMPLATE = Environment(autoescape=True).from_string(
    """<!doctype html>
<html lang="zh-Hant-TW">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Market Slice 研究報告</title>
<style>
:root{color-scheme:dark;--bg:#08121e;--surface:#0e1b2a;--border:#26384b;--text:#e8eef6;--muted:#94a8bc;--accent:#4dc9e6;--warning:#f0b429}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:system-ui,-apple-system,"Noto Sans TC",sans-serif;line-height:1.55}main{max-width:1180px;margin:auto;padding:36px 24px 64px}header{border-bottom:1px solid var(--border);padding-bottom:20px}h1{margin:0 0 8px;font-size:30px}h2{margin:0 0 16px;font-size:18px}p{margin:6px 0}.muted{color:var(--muted)}section{margin-top:28px;padding:20px 0;border-bottom:1px solid var(--border)}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px 12px;text-align:left;border-bottom:1px solid var(--border)}th{color:var(--muted);font-weight:600}.metric{font-family:ui-monospace,SFMono-Regular,Consolas,monospace}.empty{color:var(--muted);padding:14px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:1px;background:var(--border);border:1px solid var(--border)}.grid>div{background:var(--surface);padding:16px}.key{color:var(--muted);font-size:12px}.value{margin-top:5px;font-weight:650;word-break:break-word}@media(max-width:700px){main{padding:24px 16px}table{display:block;overflow:auto;white-space:nowrap}}
</style>
</head>
<body><main>
<header><h1>Market Slice 研究報告</h1><p class="muted">建立時間 {{ created_at }} · Run {{ run_id }}</p></header>
<section><h2>環境摘要</h2><div class="grid">{% for key,value in environment.items() %}<div><div class="key">{{ key }}</div><div class="value">{{ value }}</div></div>{% endfor %}</div></section>
<section><h2>Dataset 摘要</h2>{% if datasets %}<table><tr><th>Dataset</th><th>Rows</th><th>建立時間 UTC</th></tr>{% for row in datasets %}<tr><td>{{ row.dataset_type }}</td><td class="metric">{{ row.row_count }}</td><td class="metric">{{ row.created_at_utc }}</td></tr>{% endfor %}</table>{% else %}<p class="empty">目前沒有 Dataset manifest。</p>{% endif %}</section>
<section><h2>Slice coverage</h2>{% if slices %}<table><tr><th>Slice</th><th>Symbols</th><th>Fingerprint</th></tr>{% for row in slices %}<tr><td>{{ row.pair.pair_id }}</td><td class="metric">{{ row.symbols|length }}</td><td class="metric">{{ row.fingerprint }}</td></tr>{% endfor %}</table>{% else %}<p class="empty">目前沒有 Slice manifest。</p>{% endif %}</section>
<section><h2>Model metrics</h2><script type="application/json" id="plotly-model-metrics">{{ plotly_json|safe }}</script>{% if model_metrics %}<table><tr><th>Model</th><th>Direction accuracy</th><th>Return MAE</th><th>Volatility MAE</th></tr>{% for name,row in model_metrics.items() %}<tr><td>{{ name }}</td><td class="metric">{{ '%.4f'|format(row.direction_accuracy) }}</td><td class="metric">{{ '%.6f'|format(row.return_mae) }}</td><td class="metric">{{ '%.6f'|format(row.volatility_mae) }}</td></tr>{% endfor %}</table>{% else %}<p class="empty">目前沒有完成的 Model run。</p>{% endif %}</section>
<section><h2>Ensemble weights</h2>{% if weights %}<table><tr><th>Model</th><th>Weight</th></tr>{% for name,weight in weights.items() %}<tr><td>{{ name }}</td><td class="metric">{{ '%.4f'|format(weight) }}</td></tr>{% endfor %}</table>{% else %}<p class="empty">目前沒有 Ensemble weights。</p>{% endif %}</section>
<section><h2>Prediction 摘要</h2>{% if predictions %}<table><tr><th>Symbol</th><th>Horizon</th><th>Direction</th><th>Expected return</th><th>Expected volatility</th><th>Confidence</th></tr>{% for row in predictions %}<tr><td>{{ row.symbol }}</td><td>{{ row.horizon }}</td><td>{{ row.direction }}</td><td class="metric">{{ row.expected_return }}</td><td class="metric">{{ row.expected_volatility }}</td><td class="metric">{{ row.confidence_score }}</td></tr>{% endfor %}</table>{% else %}<p class="empty">目前沒有 Prediction。</p>{% endif %}</section>
</main></body></html>"""
)


def write_html_report(
    path: str | Path,
    predictions: list[PredictionRecord],
    weights: dict[str, float],
    run_id: str = "offline-smoke",
    environment: dict[str, object] | None = None,
    datasets: list[dict[str, Any]] | None = None,
    slices: list[dict[str, Any]] | None = None,
    model_metrics: dict[str, dict[str, float]] | None = None,
    **_legacy: object,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metrics = model_metrics or {}
    plotly_payload = {
        "data": [
            {
                "type": "bar",
                "name": name,
                "x": ["direction_accuracy", "return_mae", "volatility_mae"],
                "y": [
                    values.get("direction_accuracy", 0.0),
                    values.get("return_mae", 0.0),
                    values.get("volatility_mae", 0.0),
                ],
            }
            for name, values in metrics.items()
        ],
        "layout": {"title": "Model metrics"},
    }
    direction_labels = {"bullish": "偏多", "neutral": "中性", "bearish": "偏空"}
    rows = [
        {
            **prediction.model_dump(mode="json"),
            "direction": direction_labels[prediction.direction],
        }
        for prediction in predictions
    ]
    rendered = TEMPLATE.render(
        created_at=datetime.now(UTC).isoformat(),
        run_id=run_id,
        environment=environment or {"執行模式": "CPU baseline / offline smoke"},
        datasets=datasets or [],
        slices=slices or [],
        model_metrics=metrics,
        plotly_json=json.dumps(plotly_payload, separators=(",", ":")),
        weights=weights,
        predictions=rows,
    )
    destination.write_text(rendered, encoding="utf-8")
    return destination
