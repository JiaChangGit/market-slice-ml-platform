"""Typer CLI backed by the same real services as the Web UI."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import typer
from rich.console import Console
from rich.table import Table

from market_slice_ml.config.settings import get_settings
from market_slice_ml.reporting.cli_report import print_prediction
from market_slice_ml.services.models import OperationReport
from market_slice_ml.services.research_service import ResearchService
from market_slice_ml.slicing.manual_slice_loader import load_train_val_pairs

app = typer.Typer(help="本機優先的多資產市場研究平台；僅供研究，不包含交易功能。")
slices_app = typer.Typer(help="管理 synchronized Slices。")
app.add_typer(slices_app, name="slices")
console = Console()


def _service() -> ResearchService:
    return ResearchService()


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _symbols(service: ResearchService, selected: list[str] | None) -> list[str]:
    if selected:
        return selected
    targets, context = service.symbols()
    return targets + context


def _print_report(report: OperationReport) -> None:
    color = {
        "success": "green",
        "partial": "yellow",
        "no_data": "yellow",
        "disabled": "yellow",
        "failed": "red",
    }[report.status.value]
    console.print(f"[{color}]{report.message}[/{color}]")
    if report.suggested_action:
        console.print(f"下一步：{report.suggested_action}")
    for warning in report.warnings:
        console.print(f"[yellow]警告：{warning}[/yellow]")
    if report.status.value in {"failed", "disabled"}:
        raise typer.Exit(1)


@app.command()
def status() -> None:
    """顯示環境與 pipeline 狀態，不進行網路請求。"""
    service = _service()
    environment = service.environment_summary()
    console.print(
        f"Python {environment['python']} · CPU baseline · "
        f"Network {'啟用' if environment['network_enabled'] else '停用'}"
    )
    table = Table("階段", "狀態", "說明")
    labels = {"ready": "就緒", "missing": "缺少", "waiting": "等待"}
    for stage in service.pipeline_stages():
        table.add_row(stage.label_zh_tw, labels[stage.status], stage.detail)
    console.print(table)


@app.command()
def probe(live: bool = typer.Option(False, help="允許檢查 live Provider 狀態。")) -> None:
    """顯示 Provider readiness；預設不連線。"""
    service = _service()
    if live and service.settings.no_network:
        console.print("[yellow]NO_NETWORK=1，已略過 live probe。[/yellow]")
    table = Table("Provider", "狀態", "Credentials", "說明")
    for item in service.provider_readiness():
        table.add_row(item.provider_id, item.status.value, item.credentials, item.message)
    console.print(table)
    console.print("readiness check 完成；未送出 live request。")


@app.command()
def fetch(
    start: str = typer.Option(..., help="開始日期或 UTC timestamp。"),
    end: str | None = typer.Option(None, help="結束日期或 UTC timestamp。"),
    symbol: list[str] | None = typer.Option(None, "--symbol", help="可重複指定 Symbol。"),
    interval: str = typer.Option("5m", help="Provider Interval，例如 5m 或 1d。"),
    provider: str = typer.Option("yfinance", help="Provider ID。"),
) -> None:
    """抓取並 append-only 儲存歷史 Bars。"""
    service = _service()
    report = service.fetch(
        _symbols(service, symbol),
        interval,
        _parse_utc(start),
        _parse_utc(end) if end else datetime.now(UTC),
        provider,
    )
    _print_report(report)


@app.command("build-canonical")
def build_canonical(
    symbol: list[str] | None = typer.Option(None, "--symbol"),
) -> None:
    service = _service()
    _print_report(service.build_canonical(_symbols(service, symbol)))


@app.command("build-features")
def build_feature_command(
    symbol: list[str] | None = typer.Option(None, "--symbol"),
) -> None:
    service = _service()
    _print_report(service.build_features(_symbols(service, symbol)))


@app.command("build-labels")
def build_label_command(
    symbol: list[str] | None = typer.Option(None, "--symbol"),
) -> None:
    service = _service()
    targets, _ = service.symbols()
    _print_report(service.build_labels(symbol or targets))


@slices_app.command("build")
def build_slices() -> None:
    _print_report(_service().build_slices())


@slices_app.command("list")
def list_slices() -> None:
    manifests = _service().slice_manifests()
    if not manifests:
        console.print("尚未建立 Slice manifests。下一步：market-ml slices build")
        return
    table = Table("Slice", "Symbols", "Fingerprint")
    for manifest in manifests:
        table.add_row(
            str(manifest["pair"]["pair_id"]),
            str(len(manifest["symbols"])),
            str(manifest["fingerprint"])[:16],
        )
    console.print(table)


@app.command()
def train(
    all_pairs: bool = typer.Option(False, help="訓練尚未完成的所有 YAML pairs。"),
    horizon: Literal["h1", "h2", "h3", "all"] = "all",
    force: bool = typer.Option(False, help="重新訓練已完成的 pair/horizon。"),
) -> None:
    if not all_pairs:
        console.print("請使用 --all-pairs；訓練 round 由 configs/train_val_pairs.yaml 管理。")
        raise typer.Exit(1)
    _print_report(_service().train(horizon, force))


@app.command()
def validate(all_pairs: bool = typer.Option(False, help="驗證所有 YAML pairs。")) -> None:
    if not all_pairs:
        console.print("請使用 --all-pairs。")
        raise typer.Exit(1)
    pairs = load_train_val_pairs()
    console.print(f"已驗證 {len(pairs)} 組 pairs；所有 validation 均未早於 training end。")


@app.command()
def predict(
    symbol: str = typer.Option(..., help="Target Symbol，例如 NQ=F。"),
    horizon: Literal["h1", "h2", "h3"] = "h1",
    output_format: Literal["table", "json"] = "table",
) -> None:
    try:
        prediction = _service().predict(symbol, horizon)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]無法產生 Prediction：{exc}[/red]")
        console.print("下一步：確認 Features、Slice 與對應 Horizon 的 Model run 已完成。")
        raise typer.Exit(1) from exc
    if output_format == "json":
        typer.echo(prediction.model_dump_json())
    else:
        print_prediction(prediction)


@app.command()
def report(out: Path = typer.Option(Path("data/reports/latest.html"))) -> None:
    _print_report(_service().generate_report(out))


@app.command()
def web(
    host: str | None = typer.Option(None, help="預設使用 WEB_HOST。"),
    port: int | None = typer.Option(None, help="預設使用 WEB_PORT。"),
) -> None:
    """啟動本機 Web UI。"""
    import uvicorn

    from market_slice_ml.web.app import create_app

    settings = get_settings()
    selected_host = host or settings.web_host
    selected_port = port or settings.web_port
    if selected_host not in {"127.0.0.1", "localhost", "::1"} and not settings.web_api_token:
        console.print("[red]非本機 host 必須設定 WEB_API_TOKEN，服務未啟動。[/red]")
        raise typer.Exit(1)
    uvicorn.run(create_app(settings), host=selected_host, port=selected_port, log_level="info")


if __name__ == "__main__":
    app()
