"""Evaluation utilities for clinical prediction models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import (
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.calibration import calibration_curve as sk_calibration_curve


def compute_auroc(y_true: NDArray, y_prob: NDArray) -> float:
    return float(roc_auc_score(y_true, y_prob))


def compute_auprc(y_true: NDArray, y_prob: NDArray) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return float(auc(recall, precision))


def find_optimal_threshold(y_true: NDArray, y_prob: NDArray) -> float:
    """Youden's J statistic: maximize sensitivity + specificity - 1."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    return float(thresholds[np.argmax(j_scores)])


def compute_f1(y_true: NDArray, y_pred: NDArray) -> float:
    return float(f1_score(y_true, y_pred, zero_division=0))


def compute_precision_recall(
    y_true: NDArray, y_pred: NDArray
) -> tuple[float, float]:
    return (
        float(precision_score(y_true, y_pred, zero_division=0)),
        float(recall_score(y_true, y_pred, zero_division=0)),
    )


def compute_all_metrics(
    y_true: NDArray, y_prob: NDArray, threshold: float | None = None
) -> dict[str, float]:
    if threshold is None:
        threshold = find_optimal_threshold(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    precision, recall = compute_precision_recall(y_true, y_pred)
    return {
        "auroc": compute_auroc(y_true, y_prob),
        "auprc": compute_auprc(y_true, y_prob),
        "f1": compute_f1(y_true, y_pred),
        "precision": precision,
        "recall": recall,
        "threshold": threshold,
    }


def compute_calibration_curve(
    y_true: NDArray,
    y_prob: NDArray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> tuple[NDArray, NDArray]:
    fraction_of_positives, mean_predicted_value = sk_calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy=strategy
    )
    return fraction_of_positives, mean_predicted_value


def temporal_cross_validation(
    df: pd.DataFrame,
    date_column: str,
    n_splits: int = 5,
    gap_days: int = 7,
) -> list[tuple[NDArray, NDArray]]:
    """Time-based expanding-window CV. No future leakage."""
    dates = df[date_column].sort_values()
    unique_dates = np.sort(dates.unique())

    split_size = len(unique_dates) // (n_splits + 1)
    splits: list[tuple[NDArray, NDArray]] = []

    for i in range(1, n_splits + 1):
        train_end_idx = split_size * i
        train_cutoff = unique_dates[train_end_idx]
        gap_cutoff = train_cutoff + pd.Timedelta(days=gap_days)

        val_end_idx = min(train_end_idx + split_size, len(unique_dates) - 1)
        val_cutoff = unique_dates[val_end_idx]

        train_mask = dates <= train_cutoff
        val_mask = (dates > gap_cutoff) & (dates <= val_cutoff)

        if val_mask.sum() == 0:
            continue

        splits.append((np.where(train_mask)[0], np.where(val_mask)[0]))

    return splits
