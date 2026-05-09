"""Calibration utilities for probabilistic classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass(frozen=True)
class CalibrationCurve:
    """Container for calibration curve values."""

    bin_edges: NDArray[np.float64]
    bin_centers: NDArray[np.float64]
    mean_predicted: NDArray[np.float64]
    fraction_positives: NDArray[np.float64]
    counts: NDArray[np.int64]

    @property
    def gap(self) -> NDArray[np.float64]:
        return np.abs(self.fraction_positives - self.mean_predicted)


def _validate_inputs(
    y_true: NDArray[np.int64] | NDArray[np.float64],
    y_prob: NDArray[np.float64],
) -> None:
    if y_true.ndim != 1 or y_prob.ndim != 1:
        raise ValueError("y_true and y_prob must be 1D arrays.")
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have equal length.")
    if len(y_true) == 0:
        raise ValueError("y_true and y_prob must be non-empty.")
    if np.any((y_prob < 0.0) | (y_prob > 1.0)):
        raise ValueError("y_prob must be in [0, 1].")


def _make_bin_edges(
    y_prob: NDArray[np.float64],
    n_bins: int,
    strategy: str,
) -> NDArray[np.float64]:
    if strategy not in {"uniform", "quantile"}:
        raise ValueError("strategy must be one of {'uniform', 'quantile'}.")

    if strategy == "uniform":
        return np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float64)

    quantiles = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float64)
    edges = np.quantile(y_prob, quantiles)
    edges[0] = 0.0
    edges[-1] = 1.0

    # If repeated quantiles collapse bins, fallback to uniform bins.
    if np.unique(edges).size < 2:
        return np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float64)

    # Ensure strictly increasing edges for stable bin assignment.
    eps = np.finfo(np.float64).eps
    for idx in range(1, len(edges)):
        if edges[idx] <= edges[idx - 1]:
            edges[idx] = min(1.0, edges[idx - 1] + eps)
    edges[-1] = 1.0
    return edges


def compute_calibration_curve(
    y_true: NDArray[np.int64] | NDArray[np.float64],
    y_prob: NDArray[np.float64],
    n_bins: int = 10,
    strategy: str = "uniform",
) -> CalibrationCurve:
    """Compute binned calibration statistics.

    Args:
        y_true: Binary labels.
        y_prob: Predicted probabilities.
        n_bins: Number of confidence bins.
        strategy: 'uniform' or 'quantile'.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1.")

    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_prob_arr = np.asarray(y_prob, dtype=np.float64)
    _validate_inputs(y_true_arr, y_prob_arr)

    edges = _make_bin_edges(y_prob_arr, n_bins=n_bins, strategy=strategy)
    bin_ids = np.searchsorted(edges, y_prob_arr, side="right") - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    mean_predicted = np.full(n_bins, np.nan, dtype=np.float64)
    fraction_positives = np.full(n_bins, np.nan, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)

    for bin_idx in range(n_bins):
        mask = bin_ids == bin_idx
        count = int(mask.sum())
        counts[bin_idx] = count
        if count == 0:
            continue
        mean_predicted[bin_idx] = float(y_prob_arr[mask].mean())
        fraction_positives[bin_idx] = float(y_true_arr[mask].mean())

    bin_centers = (edges[:-1] + edges[1:]) / 2.0
    return CalibrationCurve(
        bin_edges=edges,
        bin_centers=bin_centers,
        mean_predicted=mean_predicted,
        fraction_positives=fraction_positives,
        counts=counts,
    )


def expected_calibration_error(
    y_true: NDArray[np.int64] | NDArray[np.float64],
    y_prob: NDArray[np.float64],
    n_bins: int = 10,
    strategy: str = "uniform",
) -> float:
    """Compute Expected Calibration Error (ECE)."""
    curve = compute_calibration_curve(
        y_true=y_true,
        y_prob=y_prob,
        n_bins=n_bins,
        strategy=strategy,
    )
    total = curve.counts.sum()
    if total == 0:
        return 0.0

    gap = np.abs(curve.fraction_positives - curve.mean_predicted)
    gap = np.nan_to_num(gap, nan=0.0)
    weights = curve.counts / total
    return float(np.sum(weights * gap))


def reliability_diagram_data(curve: CalibrationCurve) -> pd.DataFrame:
    """Return reliability diagram points as a DataFrame."""
    df = pd.DataFrame(
        {
            "bin": np.arange(len(curve.counts), dtype=np.int64),
            "bin_left": curve.bin_edges[:-1],
            "bin_right": curve.bin_edges[1:],
            "bin_center": curve.bin_centers,
            "count": curve.counts,
            "mean_predicted": curve.mean_predicted,
            "fraction_positives": curve.fraction_positives,
            "gap": curve.gap,
        }
    )
    return df


def plot_reliability_diagram(
    curve: CalibrationCurve,
    *,
    ax: Any = None,
    title: str = "Reliability Diagram",
):
    """Plot a reliability diagram.

    This helper requires matplotlib and returns `(fig, ax)`.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        ) from exc

    data = reliability_diagram_data(curve)
    bar_width = np.diff(curve.bin_edges) * 0.9

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1, label="ideal")
    ax.bar(
        data["bin_center"],
        data["fraction_positives"],
        width=bar_width,
        alpha=0.6,
        edgecolor="black",
        linewidth=0.5,
        label="observed",
    )
    ax.plot(
        data["bin_center"],
        data["mean_predicted"],
        marker="o",
        linestyle="-",
        linewidth=1,
        label="predicted",
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.25)

    return fig, ax
