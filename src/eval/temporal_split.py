"""Temporal cross-validation utilities.

This module provides sklearn-compatible time-based cross validation with
expanding and sliding windows.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import BaseCrossValidator


def _validate_n_splits(n_splits: int) -> None:
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1.")


def _validate_non_negative(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0.")


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0.")


def _coerce_dates(dates: Iterable[Any]) -> pd.Series:
    date_series = pd.to_datetime(pd.Series(dates), errors="coerce")
    if date_series.isna().any():
        raise ValueError("All dates must be parseable datetimes.")
    return date_series


def _sorted_time_index(dates: pd.Series) -> NDArray[np.int64]:
    return np.argsort(dates.to_numpy(), kind="mergesort")


def expanding_window_split(
    dates: Iterable[Any],
    n_splits: int = 5,
    *,
    min_train_size: int | None = None,
    test_size: int | None = None,
    gap: int = 0,
) -> list[tuple[NDArray[np.int64], NDArray[np.int64]]]:
    """Generate expanding-window time splits.

    Args:
        dates: Sequence of datetime-like values for each sample.
        n_splits: Number of folds.
        min_train_size: Initial train window size in rows.
        test_size: Validation window size in rows.
        gap: Number of rows between train and validation windows.
    """
    _validate_n_splits(n_splits)
    _validate_non_negative("gap", gap)

    date_series = _coerce_dates(dates)
    n_samples = len(date_series)
    if n_samples < 2:
        raise ValueError("Need at least 2 samples for time-based splitting.")

    if min_train_size is None:
        min_train_size = max(1, n_samples // (n_splits + 1))
    _validate_positive("min_train_size", min_train_size)

    if test_size is None:
        usable = n_samples - min_train_size - gap
        if usable <= 0:
            raise ValueError(
                "Not enough samples after applying min_train_size and gap."
            )
        test_size = max(1, usable // n_splits)
    _validate_positive("test_size", test_size)

    order = _sorted_time_index(date_series)
    splits: list[tuple[NDArray[np.int64], NDArray[np.int64]]] = []

    for split_id in range(n_splits):
        train_end = min_train_size + split_id * test_size
        test_start = train_end + gap
        test_end = test_start + test_size
        if test_end > n_samples:
            break

        train_idx = order[:train_end]
        test_idx = order[test_start:test_end]
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        splits.append((train_idx, test_idx))

    if not splits:
        raise ValueError(
            "No valid expanding-window splits were created. "
            "Adjust min_train_size/test_size/gap."
        )
    return splits


def sliding_window_split(
    dates: Iterable[Any],
    n_splits: int = 5,
    *,
    train_size: int,
    test_size: int | None = None,
    gap: int = 0,
) -> list[tuple[NDArray[np.int64], NDArray[np.int64]]]:
    """Generate sliding-window time splits with fixed train window size."""
    _validate_n_splits(n_splits)
    _validate_positive("train_size", train_size)
    _validate_non_negative("gap", gap)

    date_series = _coerce_dates(dates)
    n_samples = len(date_series)
    if n_samples < 2:
        raise ValueError("Need at least 2 samples for time-based splitting.")

    if test_size is None:
        usable = n_samples - train_size - gap
        if usable <= 0:
            raise ValueError("Not enough samples for requested train_size and gap.")
        test_size = max(1, usable // n_splits)
    _validate_positive("test_size", test_size)

    order = _sorted_time_index(date_series)
    splits: list[tuple[NDArray[np.int64], NDArray[np.int64]]] = []

    for split_id in range(n_splits):
        train_start = split_id * test_size
        train_end = train_start + train_size
        test_start = train_end + gap
        test_end = test_start + test_size
        if test_end > n_samples:
            break

        train_idx = order[train_start:train_end]
        test_idx = order[test_start:test_end]
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        splits.append((train_idx, test_idx))

    if not splits:
        raise ValueError(
            "No valid sliding-window splits were created. "
            "Adjust train_size/test_size/gap."
        )
    return splits


class TimeBasedSplit(BaseCrossValidator):
    """Sklearn-compatible temporal cross-validator.

    Modes:
    - "expanding": train set grows every split.
    - "sliding": fixed-size train window slides over time.
    """

    def __init__(
        self,
        n_splits: int = 5,
        *,
        date_col: str | None = None,
        mode: str = "expanding",
        min_train_size: int | None = None,
        train_size: int | None = None,
        test_size: int | None = None,
        gap: int = 0,
    ) -> None:
        if mode not in {"expanding", "sliding"}:
            raise ValueError("mode must be one of {'expanding', 'sliding'}.")

        self.n_splits = n_splits
        self.date_col = date_col
        self.mode = mode
        self.min_train_size = min_train_size
        self.train_size = train_size
        self.test_size = test_size
        self.gap = gap

    def get_n_splits(
        self,
        X: Any = None,
        y: Any = None,
        groups: Any = None,
    ) -> int:
        return self.n_splits

    def split(
        self,
        X: Any,
        y: Any = None,
        groups: Any = None,
    ):
        dates = self._resolve_dates(X, groups)

        if self.mode == "expanding":
            splits = expanding_window_split(
                dates,
                n_splits=self.n_splits,
                min_train_size=self.min_train_size,
                test_size=self.test_size,
                gap=self.gap,
            )
        else:
            if self.train_size is None:
                raise ValueError("train_size must be provided for sliding mode.")
            splits = sliding_window_split(
                dates,
                n_splits=self.n_splits,
                train_size=self.train_size,
                test_size=self.test_size,
                gap=self.gap,
            )

        for train_idx, test_idx in splits:
            yield train_idx, test_idx

    def _resolve_dates(self, X: Any, groups: Any = None) -> pd.Series:
        if groups is not None:
            return _coerce_dates(groups)

        if self.date_col is not None:
            if not isinstance(X, pd.DataFrame):
                raise ValueError("X must be a pandas DataFrame when date_col is set.")
            if self.date_col not in X.columns:
                raise ValueError(f"date_col '{self.date_col}' was not found in X.")
            return _coerce_dates(X[self.date_col])

        if isinstance(X, pd.DataFrame):
            raise ValueError(
                "Provide date_col for DataFrame input, or pass dates via groups."
            )

        return _coerce_dates(X)
