"""Persisted universal-model training and real local inference."""

from __future__ import annotations

import json
import pickle
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import numpy as np
import polars as pl
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from market_slice_ml.config.loader import flatten_symbol_config, load_yaml
from market_slice_ml.datasets.graph_dataset_builder import build_graph_dataset
from market_slice_ml.datasets.sequence_dataset_builder import (
    SequenceTensors,
    build_sequence_dataset,
)
from market_slice_ml.datasets.tabular_dataset_builder import build_tabular_dataset
from market_slice_ml.domain.enums import OperationStatus
from market_slice_ml.domain.models import PredictionRecord
from market_slice_ml.ml.baseline.tree_trainer import (
    TreeModelBundle,
    export_lgbm_to_onnx,
    train_tree_models,
)
from market_slice_ml.ml.calibration.confidence_calibrator import ConfidenceCalibrator
from market_slice_ml.ml.calibration.probability_calibrator import (
    ProbabilityCalibrator,
    save_calibrator,
)
from market_slice_ml.ml.calibration.volatility_calibrator import VolatilityCalibrator
from market_slice_ml.ml.ensemble.dynamic_ensemble_weighting import normalize_weights
from market_slice_ml.ml.ensemble.prediction_ensemble import ModelPrediction, ensemble_prediction
from market_slice_ml.ml.graph.gnn_trainer import export_gnn_to_onnx, train_gnn
from market_slice_ml.ml.graph.temporal_gnn_model import UniversalTemporalGNNModel
from market_slice_ml.ml.sequence.lstm_model import UniversalLSTMModel
from market_slice_ml.ml.sequence.lstm_trainer import export_lstm_to_onnx, train_lstm
from market_slice_ml.relationships.asset_graph import build_asset_graph
from market_slice_ml.relationships.static_weight_loader import load_relationships
from market_slice_ml.services.models import OperationReport
from market_slice_ml.storage.metadata_store import MetadataStore
from market_slice_ml.storage.path_resolver import PathResolver

Horizon = Literal["h1", "h2", "h3"]


class ModelService:
    def __init__(self, data_root: str | Path, metadata: MetadataStore) -> None:
        self.data_root = Path(data_root)
        self.metadata = metadata
        self.resolver = PathResolver(self.data_root)
        self.config = load_yaml("configs/ml.yaml")
        symbol_config = load_yaml("configs/symbols.yaml")
        self.target_symbols, self.context_symbols = flatten_symbol_config(symbol_config)
        self.feature_columns = list(self.config["datasets"]["feature_columns"])

    def train_all_pairs(
        self, horizon: Horizon | Literal["all"] = "all", force: bool = False
    ) -> OperationReport:
        horizons: list[Horizon] = ["h1", "h2", "h3"] if horizon == "all" else [horizon]
        manifests = sorted((self.data_root / "slices").glob("*/manifest.json"))
        if not manifests:
            return OperationReport(
                status=OperationStatus.FAILED,
                message="找不到可訓練的 Slice manifests。",
                suggested_action="先執行 slices build。",
            )
        completed = {
            item["training_key"]
            for item in self.metadata.list_registry("training_run")
            if item.get("status") == "completed"
        }
        runs: list[dict[str, Any]] = []
        warnings: list[str] = []
        for manifest_path in manifests:
            pair_id = manifest_path.parent.name
            for selected_horizon in horizons:
                key = f"{pair_id}:{selected_horizon}"
                if key in completed and not force:
                    warnings.append(f"{key}: 已完成，略過重複訓練")
                    continue
                try:
                    run = self._train_pair(pair_id, selected_horizon)
                    runs.append(run)
                    self.metadata.save_registry("training_run", run["run_id"], run)
                except Exception as exc:
                    warnings.append(f"{key}: {type(exc).__name__}: {exc}")
        if not runs:
            status = OperationStatus.NO_DATA if completed else OperationStatus.FAILED
            return OperationReport(
                status=status,
                message="沒有建立新的 Model artifacts。",
                suggested_action=(
                    "所有 pair 已完成；若確定要重訓請使用 --force。"
                    if completed
                    else "查看 Job log，確認 Slice coverage 與 Labels 是否足夠。"
                ),
                warnings=tuple(warnings),
            )
        return OperationReport(
            status=OperationStatus.PARTIAL if warnings else OperationStatus.SUCCESS,
            message=f"已完成 {len(runs)} 組 Model runs。",
            suggested_action="可執行 predict 或開啟 Models 頁面檢視 metrics。",
            warnings=tuple(warnings),
            result={"runs": runs},
        )

    def predict(self, symbol: str, horizon: Horizon) -> PredictionRecord:
        symbol = symbol.strip().upper()
        run = self._latest_run(horizon)
        if symbol not in run["target_symbols"]:
            raise ValueError(f"Symbol {symbol} was not a target in the latest {horizon} run")
        run_root = Path(run["run_root"])
        feature_columns = list(run["feature_columns"])
        target_mapping = {name: index for index, name in enumerate(run["target_symbols"])}
        universe_mapping = {name: index for index, name in enumerate(run["universe_symbols"])}

        target_frame = self._load_features(symbol)
        if target_frame.is_empty():
            raise FileNotFoundError(f"No Features found for {symbol}; run build-features first")
        latest = self._matrix(target_frame.tail(1), feature_columns)

        tree: TreeModelBundle = self._load_pickle(run_root / "tree.pkl")
        tree_probabilities = tree.direction.predict_proba(latest)[0]
        tree_prediction = ModelPrediction(
            tree_probabilities,
            float(tree.forward_return.predict(latest)[0]),
            float(tree.forward_volatility.predict(latest)[0]),
        )

        lstm_payload = torch.load(run_root / "lstm.pt", map_location="cpu", weights_only=True)
        lstm = UniversalLSTMModel(**lstm_payload["model_config"])
        lstm.load_state_dict(lstm_payload["state_dict"])
        sequence_length = int(lstm_payload["sequence_length"])
        if target_frame.height < sequence_length:
            raise ValueError(f"{symbol} needs {sequence_length} feature rows for LSTM inference")
        sequence = torch.from_numpy(
            self._matrix(target_frame.tail(sequence_length), feature_columns)
        ).unsqueeze(0)
        with torch.inference_mode():
            lstm_output = lstm(sequence, torch.tensor([target_mapping[symbol]]))
        lstm_prediction = ModelPrediction(
            torch.softmax(lstm_output["direction"], dim=-1).numpy()[0].astype(np.float64),
            float(lstm_output["return"].item()),
            float(lstm_output["volatility"].item()),
        )

        gnn_payload = torch.load(run_root / "gnn.pt", map_location="cpu", weights_only=True)
        gnn = UniversalTemporalGNNModel(**gnn_payload["model_config"])
        gnn.load_state_dict(gnn_payload["state_dict"])
        universe = list(run["universe_symbols"])
        node_rows: list[NDArray[np.float32]] = []
        for item in universe:
            frame = self._load_features(item)
            if frame.is_empty():
                raise FileNotFoundError(f"No Features found for graph node {item}")
            node_rows.append(self._matrix(frame.tail(1), feature_columns)[0])
        adjacency = torch.tensor(
            build_asset_graph(universe, self._relationship_weights()).adjacency,
            dtype=torch.float32,
        )
        with torch.inference_mode():
            gnn_output = gnn(
                torch.from_numpy(np.stack(node_rows)),
                torch.arange(len(universe)),
                adjacency,
            )
        node_index = universe_mapping[symbol]
        gnn_prediction = ModelPrediction(
            torch.softmax(gnn_output["direction"][node_index], dim=-1).numpy().astype(np.float64),
            float(gnn_output["return"][node_index].item()),
            float(gnn_output["volatility"][node_index].item()),
        )

        probability: ProbabilityCalibrator = self._load_pickle(
            run_root / "probability_calibrator.pkl"
        )
        volatility: VolatilityCalibrator = self._load_pickle(run_root / "volatility_calibrator.pkl")
        confidence: ConfidenceCalibrator = self._load_pickle(run_root / "confidence_calibrator.pkl")
        calibrated: dict[str, ModelPrediction] = {}
        for name, value in {
            "gbm": tree_prediction,
            "lstm": lstm_prediction,
            "gnn": gnn_prediction,
        }.items():
            calibrated[name] = ModelPrediction(
                probability.transform(np.log(value.direction_probabilities[None, :] + 1e-12))[0],
                value.expected_return,
                float(volatility.transform(np.asarray([value.expected_volatility]))[0]),
            )
        weights = json.loads((run_root / "ensemble_weights.json").read_text(encoding="utf-8"))
        prediction = ensemble_prediction(symbol, horizon, calibrated, weights)
        calibrated_confidence = 100.0 * confidence.transform(prediction.confidence_score / 100.0)
        result = prediction.model_copy(update={"confidence_score": calibrated_confidence})
        self._save_prediction(result, run)
        return result

    def list_runs(self) -> list[dict[str, Any]]:
        return sorted(
            self.metadata.list_registry("training_run"),
            key=lambda item: str(item.get("created_at_utc", "")),
            reverse=True,
        )

    def _train_pair(self, pair_id: str, horizon: Horizon) -> dict[str, Any]:
        train_frames = self._load_slice_split(pair_id, "train")
        val_frames = self._load_slice_split(pair_id, "val")
        target_train = {
            name: train_frames[name] for name in self.target_symbols if name in train_frames
        }
        target_val = {name: val_frames[name] for name in self.target_symbols if name in val_frames}
        if not target_train or not target_val:
            raise ValueError("slice has no usable target train/validation frames")
        symbol_mapping = {name: index for index, name in enumerate(sorted(target_train))}
        train_table = self._usable_table(
            build_tabular_dataset(target_train, symbol_mapping), horizon
        )
        val_table = self._usable_table(build_tabular_dataset(target_val, symbol_mapping), horizon)
        if train_table.height < 12 or val_table.height < 3:
            raise ValueError("slice does not contain enough labeled rows")
        x_train = self._matrix(train_table, self.feature_columns)
        x_val = self._matrix(val_table, self.feature_columns)
        y_train = self._directions(train_table, horizon)
        y_val = self._directions(val_table, horizon)
        return_train = self._target(train_table, f"forward_return_{horizon}")
        return_val = self._target(val_table, f"forward_return_{horizon}")
        vol_train = self._target(train_table, f"forward_volatility_{horizon}")
        vol_val = self._target(val_table, f"forward_volatility_{horizon}")
        sample_weight = self._target(train_table, "sample_weight")

        tree_estimators = int(self.config["models"]["tree"]["n_estimators"])
        tree = train_tree_models(
            x_train,
            y_train,
            return_train,
            vol_train,
            sample_weight,
            n_estimators=tree_estimators,
        )
        tree_probabilities = tree.direction.predict_proba(x_val)
        tree_scores = self._scores(
            tree_probabilities,
            tree.forward_return.predict(x_val),
            tree.forward_volatility.predict(x_val),
            y_val,
            return_val.astype(np.float64),
            vol_val.astype(np.float64),
        )

        sequence_length = int(self.config["datasets"]["sequence_length"])
        sequence_train = build_sequence_dataset(
            target_train, self.feature_columns, horizon, sequence_length
        )
        sequence_val = build_sequence_dataset(
            target_val, self.feature_columns, horizon, sequence_length
        )
        if sequence_train.features.shape[0] == 0 or sequence_val.features.shape[0] == 0:
            raise ValueError("slice does not contain enough sequence rows")
        lstm_config: dict[str, Any] = {
            "input_size": len(self.feature_columns),
            "hidden_size": int(self.config["models"]["lstm"]["hidden_size"]),
            "n_layers": int(self.config["models"]["lstm"]["n_layers"]),
            "dropout": float(self.config["models"]["lstm"]["dropout"]),
            "n_symbols": len(symbol_mapping),
            "embed_dim": int(self.config["models"]["asset_embedding_dim"]),
        }
        lstm = UniversalLSTMModel(**lstm_config)
        train_lstm(
            lstm,
            DataLoader(self._tensor_dataset(sequence_train), batch_size=64, shuffle=True),
            epochs=int(self.config["models"]["lstm"]["epochs"]),
        )
        lstm_probabilities, lstm_returns, lstm_volatility = self._lstm_predictions(
            lstm, sequence_val
        )
        lstm_scores = self._scores(
            lstm_probabilities,
            lstm_returns,
            lstm_volatility,
            sequence_val.direction.numpy(),
            sequence_val.forward_return.numpy().astype(np.float64),
            sequence_val.forward_volatility.numpy().astype(np.float64),
        )

        universe = sorted(train_frames)
        relationships = self._relationship_weights()
        adjacency = torch.tensor(
            build_asset_graph(universe, relationships).adjacency, dtype=torch.float32
        )
        graph_train = build_graph_dataset(train_frames, self.feature_columns, adjacency, horizon)
        graph_val = build_graph_dataset(val_frames, self.feature_columns, adjacency, horizon)
        if not graph_train or not graph_val:
            raise ValueError("slice has no synchronized labeled graph snapshots")
        gnn_config = {
            "node_feature_dim": len(self.feature_columns),
            "hidden_dim": int(self.config["models"]["gnn"]["hidden_dim"]),
            "heads": int(self.config["models"]["gnn"]["heads"]),
            "n_symbols": len(universe),
            "embed_dim": int(self.config["models"]["asset_embedding_dim"]),
        }
        gnn = UniversalTemporalGNNModel(**gnn_config)
        train_gnn(
            gnn,
            graph_train,
            epochs=int(self.config["models"]["gnn"]["epochs"]),
        )
        gnn_probabilities, gnn_returns, gnn_volatility, gnn_actual = self._gnn_predictions(
            gnn, graph_val
        )
        gnn_scores = self._scores(
            gnn_probabilities,
            gnn_returns,
            gnn_volatility,
            gnn_actual[0],
            gnn_actual[1],
            gnn_actual[2],
        )

        weights = normalize_weights(
            {
                "gbm": tree_scores["direction_accuracy"],
                "lstm": lstm_scores["direction_accuracy"],
                "gnn": gnn_scores["direction_accuracy"],
            }
        )
        probability = ProbabilityCalibrator().fit(np.log(tree_probabilities + 1e-12), y_val)
        volatility_calibrator = VolatilityCalibrator().fit(
            tree.forward_volatility.predict(x_val), vol_val.astype(np.float64)
        )
        confidence = ConfidenceCalibrator().fit(
            tree_probabilities.max(axis=1),
            (tree_probabilities.argmax(axis=1) == y_val),
        )

        run_id = str(uuid4())
        run_root = self.resolver.model_run(run_id)
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "tree.pkl").write_bytes(pickle.dumps(tree))
        torch.save(
            {
                "state_dict": lstm.state_dict(),
                "model_config": lstm_config,
                "sequence_length": sequence_length,
            },
            run_root / "lstm.pt",
        )
        torch.save(
            {"state_dict": gnn.state_dict(), "model_config": gnn_config},
            run_root / "gnn.pt",
        )
        (run_root / "ensemble_weights.json").write_text(
            json.dumps(weights, indent=2), encoding="utf-8"
        )
        calibration_meta = {
            "run_id": run_id,
            "dataset_version_id": pair_id,
        }
        save_calibrator(
            probability,
            run_root / "probability_calibrator.pkl",
            model_id="probability_calibrator",
            calibration_method="temperature_scaling",
            val_metrics=tree_scores,
            **calibration_meta,
        )
        save_calibrator(
            volatility_calibrator,
            run_root / "volatility_calibrator.pkl",
            model_id="volatility_calibrator",
            calibration_method="log_bias",
            val_metrics=tree_scores,
            **calibration_meta,
        )
        save_calibrator(
            confidence,
            run_root / "confidence_calibrator.pkl",
            model_id="confidence_calibrator",
            calibration_method="accuracy_scaling",
            val_metrics=tree_scores,
            **calibration_meta,
        )
        self._export_onnx(tree, lstm, gnn, run_root, sequence_length, len(universe))

        run = {
            "run_id": run_id,
            "training_key": f"{pair_id}:{horizon}",
            "pair_id": pair_id,
            "horizon": horizon,
            "status": "completed",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "run_root": str(run_root),
            "feature_columns": self.feature_columns,
            "target_symbols": sorted(target_train),
            "universe_symbols": universe,
            "weights": weights,
            "metrics": {"gbm": tree_scores, "lstm": lstm_scores, "gnn": gnn_scores},
        }
        (run_root / "run_manifest.json").write_text(
            json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return run

    def _export_onnx(
        self,
        tree: TreeModelBundle,
        lstm: UniversalLSTMModel,
        gnn: UniversalTemporalGNNModel,
        root: Path,
        sequence_length: int,
        n_nodes: int,
    ) -> None:
        onnx_root = root / "onnx"
        export_lgbm_to_onnx(
            tree.direction.model.booster_,
            len(self.feature_columns),
            onnx_root / "tree_direction.onnx",
        )
        export_lgbm_to_onnx(
            tree.forward_return.model.booster_,
            len(self.feature_columns),
            onnx_root / "tree_return.onnx",
        )
        export_lgbm_to_onnx(
            tree.forward_volatility.model.booster_,
            len(self.feature_columns),
            onnx_root / "tree_volatility.onnx",
        )
        export_lstm_to_onnx(
            lstm, sequence_length, len(self.feature_columns), onnx_root / "lstm.onnx"
        )
        export_gnn_to_onnx(gnn, n_nodes, len(self.feature_columns), onnx_root / "gnn.onnx")

    def _latest_run(self, horizon: Horizon) -> dict[str, Any]:
        runs = [
            item
            for item in self.list_runs()
            if item.get("horizon") == horizon and item.get("status") == "completed"
        ]
        if not runs:
            raise FileNotFoundError(f"No completed Model run found for {horizon}")
        return runs[0]

    def _load_slice_split(self, pair_id: str, split: str) -> dict[str, pl.DataFrame]:
        root = self.data_root / "slices" / pair_id / split
        frames: dict[str, pl.DataFrame] = {}
        for path in sorted(root.glob("*.parquet")):
            frame = pl.read_parquet(path)
            if frame.is_empty() or "symbol" not in frame.columns:
                continue
            symbols = frame.get_column("symbol").unique().to_list()
            if len(symbols) != 1:
                raise ValueError(f"Slice file must contain exactly one Symbol: {path}")
            frames[str(symbols[0])] = frame
        return frames

    def _load_features(self, symbol: str) -> pl.DataFrame:
        safe = self.resolver.safe_component(symbol)
        paths = sorted((self.data_root / "features" / safe).rglob("*.parquet"))
        return pl.read_parquet(paths).sort("timestamp_utc") if paths else pl.DataFrame()

    def _relationship_weights(self) -> dict[tuple[str, str], float]:
        return {(item.source, item.target): item.static_weight for item in load_relationships()}

    def _usable_table(self, frame: pl.DataFrame, horizon: Horizon) -> pl.DataFrame:
        required = [
            f"direction_{horizon}",
            f"forward_return_{horizon}",
            f"forward_volatility_{horizon}",
        ]
        missing = set(self.feature_columns + required) - set(frame.columns)
        if missing:
            raise ValueError(f"training data is missing columns: {sorted(missing)}")
        return frame.filter(pl.all_horizontal(pl.col(name).is_not_null() for name in required))

    def _matrix(self, frame: pl.DataFrame, columns: list[str]) -> NDArray[np.float32]:
        missing = set(columns) - set(frame.columns)
        if missing:
            raise ValueError(f"Features 缺少 Model 所需欄位：{sorted(missing)}")
        return (
            frame.select(columns)
            .with_columns(pl.all().cast(pl.Float32, strict=False))
            .fill_nan(None)
            .fill_null(0.0)
            .to_numpy()
            .astype(np.float32)
        )

    @staticmethod
    def _directions(frame: pl.DataFrame, horizon: Horizon) -> NDArray[np.int64]:
        mapping = {"bearish": 0, "neutral": 1, "bullish": 2}
        return np.asarray(
            [mapping[str(value)] for value in frame.get_column(f"direction_{horizon}")],
            dtype=np.int64,
        )

    @staticmethod
    def _target(frame: pl.DataFrame, column: str) -> NDArray[np.float32]:
        return frame.get_column(column).to_numpy().astype(np.float32)

    @staticmethod
    def _tensor_dataset(tensors: SequenceTensors) -> TensorDataset:
        return TensorDataset(
            tensors.features,
            tensors.symbol_idx,
            tensors.direction,
            tensors.forward_return,
            tensors.forward_volatility,
        )

    @staticmethod
    def _lstm_predictions(
        model: UniversalLSTMModel, tensors: SequenceTensors
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        model.eval()
        with torch.inference_mode():
            output = model(tensors.features, tensors.symbol_idx)
        return (
            torch.softmax(output["direction"], dim=-1).numpy().astype(np.float64),
            output["return"].numpy().astype(np.float64),
            output["volatility"].numpy().astype(np.float64),
        )

    @staticmethod
    def _gnn_predictions(
        model: UniversalTemporalGNNModel,
        snapshots: list[dict[str, Tensor]],
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]],
    ]:
        probabilities: list[NDArray[np.float64]] = []
        returns: list[NDArray[np.float64]] = []
        volatilities: list[NDArray[np.float64]] = []
        actual_direction: list[NDArray[np.int64]] = []
        actual_return: list[NDArray[np.float64]] = []
        actual_volatility: list[NDArray[np.float64]] = []
        model.eval()
        with torch.inference_mode():
            for snapshot in snapshots:
                mask = snapshot["target_mask"]
                output = model(
                    snapshot["node_features"],
                    snapshot["symbol_idx"],
                    snapshot["adjacency_matrix"],
                )
                probabilities.append(
                    torch.softmax(output["direction"][mask], dim=-1).numpy().astype(np.float64)
                )
                returns.append(output["return"][mask].numpy().astype(np.float64))
                volatilities.append(output["volatility"][mask].numpy().astype(np.float64))
                actual_direction.append(snapshot["direction"][mask].numpy().astype(np.int64))
                actual_return.append(snapshot["forward_return"][mask].numpy().astype(np.float64))
                actual_volatility.append(
                    snapshot["forward_volatility"][mask].numpy().astype(np.float64)
                )
        return (
            np.concatenate(probabilities),
            np.concatenate(returns),
            np.concatenate(volatilities),
            (
                np.concatenate(actual_direction),
                np.concatenate(actual_return),
                np.concatenate(actual_volatility),
            ),
        )

    @staticmethod
    def _scores(
        probabilities: NDArray[np.float64],
        predicted_return: NDArray[np.float64],
        predicted_volatility: NDArray[np.float64],
        actual_direction: NDArray[np.int64],
        actual_return: NDArray[np.float64],
        actual_volatility: NDArray[np.float64],
    ) -> dict[str, float]:
        return {
            "direction_accuracy": float(np.mean(probabilities.argmax(axis=1) == actual_direction)),
            "return_mae": float(np.mean(np.abs(predicted_return - actual_return))),
            "volatility_mae": float(np.mean(np.abs(predicted_volatility - actual_volatility))),
        }

    @staticmethod
    def _load_pickle(path: Path) -> Any:
        return pickle.loads(path.read_bytes())

    def _save_prediction(self, prediction: PredictionRecord, run: dict[str, Any]) -> None:
        root = self.data_root / "predictions" / str(run["run_id"])
        root.mkdir(parents=True, exist_ok=True)
        filename = f"{self.resolver.safe_component(prediction.symbol)}_{prediction.horizon}.json"
        public_path = root / filename
        public_path.write_text(prediction.model_dump_json(indent=2), encoding="utf-8")
        audit = {
            "run_id": run["run_id"],
            "pair_id": run["pair_id"],
            "created_at_utc": datetime.now(UTC).isoformat(),
            "public_prediction_path": str(public_path),
        }
        (root / f"{public_path.stem}_audit.json").write_text(
            json.dumps(audit, indent=2), encoding="utf-8"
        )
