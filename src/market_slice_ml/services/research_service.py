"""Real local data operations shared by CLI, jobs, and Web API."""

from __future__ import annotations

import json
import platform
import sys
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

import polars as pl

from market_slice_ml.config.loader import flatten_symbol_config, load_yaml
from market_slice_ml.config.settings import Settings, get_settings
from market_slice_ml.domain.enums import DatasetType, OperationStatus
from market_slice_ml.domain.models import PredictionRecord
from market_slice_ml.features.feature_builder import build_features
from market_slice_ml.labels.label_builder import build_labels
from market_slice_ml.processing.canonical_builder import build_canonical_bars
from market_slice_ml.providers.base import BaseProvider, ProviderFetchResult
from market_slice_ml.providers.provider_registry import ProviderRegistry
from market_slice_ml.relationships.static_weight_loader import load_relationships
from market_slice_ml.reporting.html_report import write_html_report
from market_slice_ml.services.model_service import Horizon, ModelService
from market_slice_ml.services.models import OperationReport, PipelineStage, ProviderReadiness
from market_slice_ml.slicing.manual_slice_loader import load_train_val_pairs
from market_slice_ml.slicing.synchronized_slice_builder import build_synchronized_slice
from market_slice_ml.storage.duckdb_store import DuckDBStore
from market_slice_ml.storage.metadata_store import MetadataStore
from market_slice_ml.storage.parquet_store import ParquetStore
from market_slice_ml.storage.path_resolver import PathResolver
from market_slice_ml.versioning.dataset_fingerprint import dataset_fingerprint
from market_slice_ml.versioning.dataset_manifest import DatasetManifest


class ResearchService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.data_root = self.settings.data_root.resolve()
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.resolver = PathResolver(self.data_root)
        self.parquet = ParquetStore()
        self.duckdb = DuckDBStore()
        self.metadata = MetadataStore(self.data_root / "metadata.db")
        self.registry = ProviderRegistry()
        self.models = ModelService(self.data_root, self.metadata)

    def symbols(self) -> tuple[list[str], list[str]]:
        return flatten_symbol_config(load_yaml("configs/symbols.yaml"))

    def environment_summary(self) -> dict[str, object]:
        torch_version: str | None = None
        gpu_available = False
        try:
            import torch

            torch_version = torch.__version__
            gpu_available = bool(torch.cuda.is_available())
        except Exception:
            torch_version = None
        return {
            "python": sys.version.split()[0],
            "python_executable": sys.executable,
            "os": platform.platform(),
            "cpu_baseline": True,
            "torch": torch_version,
            "gpu_available": gpu_available,
            "network_enabled": not self.settings.no_network,
            "data_root": str(self.data_root),
        }

    def pipeline_stages(self) -> list[PipelineStage]:
        checks = [
            ("raw", "原始 Bars", self.data_root / "raw"),
            ("canonical", "Canonical 5m", self.data_root / "canonical"),
            ("features", "Features", self.data_root / "features"),
            ("labels", "Labels", self.data_root / "labels"),
            ("slices", "Slices", self.data_root / "slices"),
            ("models", "Models", self.data_root / "models" / "runs"),
            ("reports", "Reports", self.data_root / "reports"),
        ]
        stages: list[PipelineStage] = []
        blocked = False
        for stage_id, label, path in checks:
            has_output = path.exists() and any(item.is_file() for item in path.rglob("*"))
            if has_output:
                status = "ready"
                detail = "已有本機產物"
            elif blocked:
                status = "waiting"
                detail = "等待前一階段"
            else:
                status = "missing"
                detail = "尚未建立"
                blocked = True
            stages.append(
                PipelineStage(
                    stage_id=stage_id,
                    label_zh_tw=label,
                    status=status,
                    detail=detail,
                )
            )
        return stages

    def provider_readiness(self) -> list[ProviderReadiness]:
        examples = {
            "stooq": ("SPY", "1d"),
            "cboe": ("^VIX", "5m"),
        }
        result: list[ProviderReadiness] = []
        for provider_id, provider in self.registry.all_providers.items():
            symbol, interval = examples.get(provider_id, ("SPY", "5m"))
            probe = provider.probe(symbol, interval)
            status = (
                OperationStatus.SUCCESS
                if probe.available
                else (
                    probe.status
                    if probe.status is not OperationStatus.SUCCESS
                    else OperationStatus.DISABLED
                )
            )
            credentials = self._credential_status(provider_id, provider)
            message, suggested_action = self._readiness_copy(provider_id, probe)
            result.append(
                ProviderReadiness(
                    provider_id=provider_id,
                    status=status,
                    credentials=credentials,
                    message=message,
                    suggested_action=suggested_action,
                )
            )
        return result

    def fetch(
        self,
        symbols: list[str],
        interval: str,
        start: datetime,
        end: datetime,
        provider_id: str = "yfinance",
    ) -> OperationReport:
        if self.settings.no_network:
            return OperationReport(
                status=OperationStatus.DISABLED,
                message="目前已啟用離線模式，未送出任何網路請求。",
                suggested_action="將 NO_NETWORK 設為 0，重新啟動服務後再執行 Fetch。",
            )
        if end <= start:
            return OperationReport(
                status=OperationStatus.FAILED,
                message="結束時間必須晚於開始時間。",
                suggested_action="修正 UTC 日期範圍後重試。",
            )
        try:
            provider = self.registry.get(provider_id)
        except ValueError:
            return OperationReport(
                status=OperationStatus.FAILED,
                message=f"找不到 Provider：{provider_id}。",
                suggested_action="執行 market-ml probe 查看可用的 Provider ID。",
            )
        results = [provider.fetch_with_status(symbol, interval, start, end) for symbol in symbols]
        paths: list[str] = []
        warnings: list[str] = []
        details: list[dict[str, object]] = []
        for fetched in results:
            details.append(self._provider_result_payload(fetched))
            if fetched.status is OperationStatus.SUCCESS:
                paths.extend(self._write_raw(fetched))
            else:
                warnings.append(f"{fetched.symbol}: {fetched.message}")
        succeeded = sum(item.status is OperationStatus.SUCCESS for item in results)
        if succeeded == 0:
            status = (
                OperationStatus.DISABLED
                if all(item.status is OperationStatus.DISABLED for item in results)
                else OperationStatus.FAILED
            )
            message = "沒有任何 Symbol 完成 Fetch。"
            action = "查看 Provider 狀態與 Job log，確認 Interval、日期範圍及憑證。"
        elif succeeded < len(results):
            status = OperationStatus.PARTIAL
            message = f"已完成 {succeeded}/{len(results)} 個 Symbol，其餘項目需要處理。"
            action = "查看警告內容後，只重試失敗的 Symbol。"
        else:
            status = OperationStatus.SUCCESS
            message = f"已完成 {succeeded} 個 Symbol 的 Fetch。"
            action = "下一步執行 build-canonical。"
        return OperationReport(
            status=status,
            message=message,
            suggested_action=action,
            warnings=tuple(warnings),
            result={"paths": paths, "providers": details},
        )

    def build_canonical(self, symbols: list[str], interval: str = "5m") -> OperationReport:
        priorities = {
            name: int(config.get("priority", 100))
            for name, config in load_yaml("configs/providers.yaml").get("providers", {}).items()
        }
        outputs: list[str] = []
        built: dict[str, pl.DataFrame] = {}
        warnings: list[str] = []
        for symbol in symbols:
            frames = self._load_raw_by_provider(symbol, interval)
            if not frames:
                warnings.append(f"{symbol}: 找不到 {interval} 原始 Bars")
                continue
            canonical = build_canonical_bars(
                frames,
                symbol,
                futures=symbol.endswith("=F"),
                priorities=priorities,
            )
            if canonical.is_empty():
                warnings.append(f"{symbol}: Canonical 結果為空")
                continue
            built[symbol] = canonical
            outputs.extend(self._write_stage_partitions("canonical", symbol, canonical))
        if outputs:
            self._register_manifest(
                DatasetType.CANONICAL_5M, built, outputs, {"interval": interval}
            )
        return self._build_report(
            "Canonical 5m", len(symbols), built, outputs, warnings, "build-features"
        )

    def build_features(self, symbols: list[str]) -> OperationReport:
        canonical = {symbol: self._load_stage("canonical", symbol) for symbol in symbols}
        canonical = {symbol: frame for symbol, frame in canonical.items() if not frame.is_empty()}
        relationships = load_relationships()
        outputs: list[str] = []
        built: dict[str, pl.DataFrame] = {}
        warnings: list[str] = []
        vix = canonical.get("^VIX")
        for symbol in symbols:
            frame = canonical.get(symbol)
            if frame is None:
                warnings.append(f"{symbol}: 找不到 Canonical Bars")
                continue
            configured = [item for item in relationships if item.target == symbol]
            context = {
                item.source: canonical[item.source]
                for item in configured
                if item.source in canonical
            }
            weights = {item.source: item.static_weight for item in configured}
            features = build_features(frame, context, weights, vix)
            built[symbol] = features
            outputs.extend(self._write_stage_partitions("features", symbol, features))
        if outputs:
            self._register_manifest(DatasetType.FEATURES, built, outputs, {"relationships": True})
        return self._build_report(
            "Features", len(symbols), built, outputs, warnings, "build-labels"
        )

    def build_labels(self, target_symbols: list[str]) -> OperationReport:
        outputs: list[str] = []
        built: dict[str, pl.DataFrame] = {}
        warnings: list[str] = []
        config = load_yaml("configs/ml.yaml")
        horizons = {name: int(value) for name, value in config.get("horizons", {}).items()}
        thresholds = config.get("labels", {})
        for symbol in target_symbols:
            features = self._load_stage("features", symbol)
            if features.is_empty():
                warnings.append(f"{symbol}: 找不到 Features")
                continue
            labels = build_labels(
                features,
                horizons=horizons,
                bullish_threshold=float(thresholds.get("bullish_threshold", 0.003)),
                bearish_threshold=float(thresholds.get("bearish_threshold", -0.003)),
            )
            built[symbol] = labels
            outputs.extend(self._write_stage_partitions("labels", symbol, labels))
        if outputs:
            self._register_manifest(DatasetType.LABELS, built, outputs, config)
        return self._build_report(
            "Labels", len(target_symbols), built, outputs, warnings, "slices build"
        )

    def build_slices(self) -> OperationReport:
        targets, context = self.symbols()
        pairs = load_train_val_pairs()
        frames: dict[str, pl.DataFrame] = {}
        for symbol in targets:
            frame = self._load_stage("labels", symbol)
            if not frame.is_empty():
                frames[symbol] = frame
        for symbol in context:
            frame = self._load_stage("features", symbol)
            if not frame.is_empty():
                frames[symbol] = frame
        if not frames:
            return OperationReport(
                status=OperationStatus.FAILED,
                message="沒有可建立 Slice 的 Labels 或 Features。",
                suggested_action="依序執行 build-canonical、build-features、build-labels。",
            )
        warnings: list[str] = []
        manifests: list[str] = []
        for pair in pairs:
            sliced = build_synchronized_slice(frames, pair)
            root = self.resolver.slice_root(pair.pair_id)
            paths: list[str] = []
            coverage: dict[str, dict[str, int]] = {}
            for split_name, split_frames in (
                ("train", sliced.train),
                ("val", sliced.validation),
            ):
                for symbol, frame in split_frames.items():
                    if frame.is_empty():
                        warnings.append(f"{pair.pair_id}/{symbol}/{split_name}: 無資料")
                        continue
                    path = root / split_name / f"{self.resolver.safe_component(symbol)}.parquet"
                    self.parquet.write_replace(frame, path)
                    paths.append(str(path))
                    coverage.setdefault(symbol, {})[split_name] = frame.height
            manifest = {
                "pair": pair.model_dump(mode="json"),
                "created_at_utc": datetime.now(UTC).isoformat(),
                "symbols": sorted(coverage),
                "coverage": coverage,
                "paths": paths,
                "fingerprint": dataset_fingerprint(
                    {"pair": pair.model_dump(mode="json"), "coverage": coverage}
                ),
            }
            manifest_path = root / "manifest.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            manifests.append(str(manifest_path))
        status = OperationStatus.PARTIAL if warnings else OperationStatus.SUCCESS
        return OperationReport(
            status=status,
            message=f"已建立 {len(manifests)} 組 synchronized Slices。",
            suggested_action="下一步執行 train --all-pairs。",
            warnings=tuple(warnings),
            result={"manifests": manifests},
        )

    def slice_manifests(self) -> list[dict[str, Any]]:
        manifests: list[dict[str, Any]] = []
        root = self.data_root / "slices"
        if not root.exists():
            return manifests
        for path in sorted(root.glob("*/manifest.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["manifest_path"] = str(path)
            manifests.append(payload)
        return manifests

    def train(
        self, horizon: Horizon | Literal["all"] = "all", force: bool = False
    ) -> OperationReport:
        return self.models.train_all_pairs(horizon, force)

    def predict(self, symbol: str, horizon: Horizon) -> PredictionRecord:
        return self.models.predict(symbol, horizon)

    def model_runs(self) -> list[dict[str, Any]]:
        return self.models.list_runs()

    def generate_report(self, out: str | Path | None = None) -> OperationReport:
        runs = self.model_runs()
        latest = runs[0] if runs else {}
        predictions: list[PredictionRecord] = []
        prediction_root = self.data_root / "predictions"
        if prediction_root.exists():
            for path in sorted(prediction_root.glob("*/*.json")):
                if not path.name.endswith("_audit.json"):
                    predictions.append(PredictionRecord.model_validate_json(path.read_text()))
        destination = Path(out) if out else self.resolver.report("latest.html")
        path = write_html_report(
            destination,
            predictions,
            dict(latest.get("weights", {})),
            run_id=str(latest.get("run_id", "尚無 Model run")),
            environment=self.environment_summary(),
            datasets=self.metadata.list_manifests(),
            slices=self.slice_manifests(),
            model_metrics=dict(latest.get("metrics", {})),
        )
        return OperationReport(
            status=OperationStatus.SUCCESS,
            message="Report 已建立。",
            suggested_action="可在 Reports 頁面開啟 self-contained HTML。",
            result={"path": str(path)},
        )

    def _write_raw(self, fetched: ProviderFetchResult) -> list[str]:
        paths: list[str] = []
        dated = fetched.frame.with_columns(
            pl.col("timestamp_utc").dt.date().alias("_partition_day")
        )
        batch = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
        for partition in dated.partition_by("_partition_day", maintain_order=True):
            day = partition.item(0, "_partition_day")
            if not isinstance(day, date):
                raise TypeError("raw partition date is invalid")
            path = self.resolver.raw(
                fetched.provider_id, fetched.symbol, fetched.interval, day, batch
            )
            self.parquet.write_append_only(partition.drop("_partition_day"), path)
            paths.append(str(path))
        return paths

    def _load_raw_by_provider(self, symbol: str, interval: str) -> list[pl.DataFrame]:
        safe = self.resolver.safe_component(symbol)
        root = self.data_root / "raw"
        if not root.exists():
            return []
        frames: list[pl.DataFrame] = []
        for provider_root in root.iterdir():
            paths = sorted((provider_root / safe).rglob(f"{safe}_{interval}_*.parquet"))
            if paths:
                frames.append(self.duckdb.read_parquet(paths))
        return frames

    def _load_stage(self, stage: str, symbol: str) -> pl.DataFrame:
        safe = self.resolver.safe_component(symbol)
        paths = sorted((self.data_root / stage / safe).rglob("*.parquet"))
        return self.duckdb.read_parquet(paths) if paths else pl.DataFrame()

    def _write_stage_partitions(self, stage: str, symbol: str, frame: pl.DataFrame) -> list[str]:
        resolver: Callable[[str, date], Path]
        if stage == "canonical":
            resolver = self.resolver.canonical
        elif stage == "features":
            resolver = self.resolver.features
        elif stage == "labels":
            resolver = self.resolver.labels
        else:
            raise ValueError(f"unsupported stage: {stage}")
        outputs: list[str] = []
        dated = frame.with_columns(pl.col("timestamp_utc").dt.date().alias("_partition_day"))
        for partition in dated.partition_by("_partition_day", maintain_order=True):
            day = partition.item(0, "_partition_day")
            if not isinstance(day, date):
                raise TypeError("derived partition date is invalid")
            path = resolver(symbol, day)
            self.parquet.write_replace(partition.drop("_partition_day"), path)
            outputs.append(str(path))
        return outputs

    def _register_manifest(
        self,
        dataset_type: DatasetType,
        frames: dict[str, pl.DataFrame],
        paths: list[str],
        config: dict[str, object],
    ) -> DatasetManifest:
        timestamps = [
            value
            for frame in frames.values()
            for value in (
                frame.get_column("timestamp_utc").min(),
                frame.get_column("timestamp_utc").max(),
            )
            if isinstance(value, datetime)
        ]
        targets, context = self.symbols()
        manifest = DatasetManifest(
            dataset_type=dataset_type,
            created_at_utc=datetime.now(UTC),
            config_hash=dataset_fingerprint(config),
            symbol_universe_hash=dataset_fingerprint(sorted(targets + context)),
            row_count=sum(frame.height for frame in frames.values()),
            timestamp_min_utc=min(timestamps) if timestamps else None,
            timestamp_max_utc=max(timestamps) if timestamps else None,
            quality_summary={
                symbol: self._quality_mean(frame)
                for symbol, frame in frames.items()
                if "quality_score" in frame.columns
            },
            parquet_paths=paths,
        )
        self.metadata.register_manifest(manifest)
        return manifest

    @staticmethod
    def _quality_mean(frame: pl.DataFrame) -> float:
        value = frame.get_column("quality_score").mean()
        return float(value) if isinstance(value, (int, float)) else 0.0

    @staticmethod
    def _credential_status(provider_id: str, provider: BaseProvider) -> str:
        if provider_id in {"yfinance", "stooq", "cboe"}:
            return "不需要"
        return "已設定" if provider.is_enabled else "未設定"

    @staticmethod
    def _readiness_copy(provider_id: str, probe: Any) -> tuple[str, str]:
        if probe.available:
            return (
                "設定可用；readiness check 未送出 live request。",
                "需要資料時，先將 NO_NETWORK 設為 0，再執行明確的 Fetch。",
            )
        disabled_messages = {
            "ibkr_realtime": "Realtime Provider 依規格停用。",
            "schwab": "Schwab Provider 依規格停用。",
            "akshare": "AKShare 預設停用。",
            "alpha_vantage": "缺少 Alpha Vantage API key，Provider 未啟用。",
            "ibkr_historical": "IBKR Historical 尚未啟用或缺少 optional library。",
        }
        actions = {
            "ibkr_realtime": "研究資料請改用 IBKR Historical 或其他 Historical Provider。",
            "schwab": "請選擇已啟用的 Historical Provider。",
            "akshare": "確定需要時，安裝 optional dependency 並設定 AKSHARE_ENABLED=true。",
            "alpha_vantage": "在 .env 設定 ALPHA_VANTAGE_API_KEY 後重新啟動。",
            "ibkr_historical": "安裝 ib_insync、啟用設定，並確認 IBKR Historical endpoint。",
        }
        return (
            disabled_messages.get(provider_id, f"Provider 無法使用：{probe.message}"),
            actions.get(provider_id, probe.suggested_action or "請檢查 Provider 設定。"),
        )

    @staticmethod
    def _provider_result_payload(result: ProviderFetchResult) -> dict[str, object]:
        return {
            "provider": result.provider_id,
            "symbol": result.symbol,
            "interval": result.interval,
            "status": result.status.value,
            "rows": result.frame.height,
            "technical_detail": result.message,
            "suggested_action": result.suggested_action,
        }

    @staticmethod
    def _build_report(
        stage: str,
        requested: int,
        built: dict[str, pl.DataFrame],
        outputs: list[str],
        warnings: list[str],
        next_command: str,
    ) -> OperationReport:
        if not built:
            return OperationReport(
                status=OperationStatus.FAILED,
                message=f"{stage} 未建立任何產物。",
                suggested_action="查看警告與 Job log，先補齊上一階段資料。",
                warnings=tuple(warnings),
            )
        status = (
            OperationStatus.PARTIAL
            if len(built) < requested or warnings
            else OperationStatus.SUCCESS
        )
        return OperationReport(
            status=status,
            message=f"{stage} 已完成 {len(built)}/{requested} 個 Symbol。",
            suggested_action=f"下一步執行 {next_command}。",
            warnings=tuple(warnings),
            result={"paths": outputs, "rows": sum(frame.height for frame in built.values())},
        )
