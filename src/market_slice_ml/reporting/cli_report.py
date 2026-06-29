"""Rich CLI rendering for six-field predictions."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from market_slice_ml.domain.models import PredictionRecord


def prediction_table(prediction: PredictionRecord) -> Table:
    table = Table(title=f"Prediction: {prediction.symbol} | {prediction.horizon}")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Direction", prediction.direction.upper())
    table.add_row("Expected return", f"{prediction.expected_return:+.2%}")
    table.add_row("Expected volatility", f"{prediction.expected_volatility:.2%}")
    table.add_row("Confidence", f"{prediction.confidence_score:.1f} / 100")
    return table


def print_prediction(prediction: PredictionRecord, console: Console | None = None) -> None:
    (console or Console()).print(prediction_table(prediction))
