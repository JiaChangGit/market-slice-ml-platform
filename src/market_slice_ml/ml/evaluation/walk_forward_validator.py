"""Leakage-guarded chronological fold construction."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from market_slice_ml.domain.exceptions import DataLeakError


@dataclass(frozen=True)
class WalkForwardFold:
    fold_id: str
    train_start: datetime
    train_end: datetime
    val_start: datetime
    val_end: datetime

    def __post_init__(self) -> None:
        if self.val_start < self.train_end:
            raise DataLeakError("validation fold overlaps training")


def validate_folds(folds: list[WalkForwardFold]) -> None:
    for fold in folds:
        if fold.val_start < fold.train_end:
            raise DataLeakError(f"fold {fold.fold_id} contains lookahead")
