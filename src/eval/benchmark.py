"""Benchmark runner — XGBoost vs FT-Transformer comparison."""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.eval_utils import (
    compute_auroc,
    compute_auprc,
    compute_f1_at_threshold,
    compute_precision_recall_at_threshold,
)
from src.eval.calibration import compute_ece, compute_mce


def run_benchmark(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    models: dict[str, object] | None = None,
    output_dir: Path = Path("results"),
) -> dict:
    """Run benchmark comparison of multiple models.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data (hyperparameter tuning)
        X_test, y_test: Test data (final evaluation)
        models: Dict of model_name -> fitted model
        output_dir: Directory to save results

    Returns:
        Dict of results per model
    """
    if models is None:
        models = {}

    results = {}
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        start_time = time.time()

        try:
            # Get predictions
            if hasattr(model, "predict_proba"):
                y_val_prob = model.predict_proba(X_val)[:, 1]
                y_test_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_val_prob = model.predict(X_val)
                y_test_prob = model.predict(X_test)

            eval_time = time.time() - start_time

            # Compute metrics
            val_auroc = compute_auroc(y_val, y_val_prob)
            val_auprc = compute_auprc(y_val, y_val_prob)
            test_auroc = compute_auroc(y_test, y_test_prob)
            test_auprc = compute_auprc(y_test, y_test_prob)

            # Find optimal threshold on validation
            optimal_threshold = find_optimal_threshold(y_val, y_val_prob)
            val_f1 = compute_f1_at_threshold(y_val, y_val_prob, optimal_threshold)
            val_prec, val_rec = compute_precision_recall_at_threshold(
                y_val, y_val_prob, optimal_threshold
            )

            test_f1 = compute_f1_at_threshold(y_test, y_test_prob, optimal_threshold)
            test_prec, test_rec = compute_precision_recall_at_threshold(
                y_test, y_test_prob, optimal_threshold
            )

            # Calibration metrics
            test_ece = compute_ece(y_test.values, y_test_prob, n_bins=10)
            test_mce = compute_mce(y_test.values, y_test_prob, n_bins=10)

            results[model_name] = {
                "test": {
                    "auroc": float(test_auroc),
                    "auprc": float(test_auprc),
                    "f1": float(test_f1),
                    "precision": float(test_prec),
                    "recall": float(test_rec),
                    "optimal_threshold": float(optimal_threshold),
                    "ece": float(test_ece),
                    "mce": float(test_mce),
                },
                "validation": {
                    "auroc": float(val_auroc),
                    "auprc": float(val_auprc),
                    "f1": float(val_f1),
                    "precision": float(val_prec),
                    "recall": float(val_rec),
                },
                "timing": {
                    "eval_time_seconds": round(eval_time, 2),
                },
                "n_train": len(y_train),
                "n_val": len(y_val),
                "n_test": len(y_test),
                "positive_rate_test": float(y_test.mean()),
            }

            print(
                f"  Test AUROC: {test_auroc:.4f} | "
                f"Test AUPRC: {test_auprc:.4f} | "
                f"Test F1: {test_f1:.4f}"
            )
            print(f"  Test ECE: {test_ece:.4f} | MCE: {test_mce:.4f}")

        except Exception as e:
            print(f"  ERROR: {e}")
            results[model_name] = {"error": str(e)}

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / f"benchmark_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_file}")

    return results


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find optimal threshold using Youden's J statistic."""
    best_j = -1
    best_thresh = 0.5

    for thresh in np.linspace(0.01, 0.99, 99):
        y_pred = (y_prob >= thresh).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        j = sensitivity + specificity - 1

        if j > best_j:
            best_j = j
            best_thresh = thresh

    return best_thresh


def compare_to_baseline(
    results: dict, baseline_model: str = "xgb_baseline"
) -> pd.DataFrame:
    """Compare all models to a baseline.

    Args:
        results: Dict from run_benchmark()
        baseline_model: Name of baseline model to compare against

    Returns:
        DataFrame with delta metrics vs baseline
    """
    if baseline_model not in results:
        raise ValueError(f"Baseline model {baseline_model} not in results")

    baseline = results[baseline_model]["test"]
    comparison = []

    for model_name, model_results in results.items():
        if "error" in model_results or model_name == baseline_model:
            continue

        test_metrics = model_results.get("test", {})
        delta = {
            "model": model_name,
            "delta_auroc": test_metrics.get("auroc", 0) - baseline.get("auroc", 0),
            "delta_auprc": test_metrics.get("auprc", 0) - baseline.get("auprc", 0),
            "delta_f1": test_metrics.get("f1", 0) - baseline.get("f1", 0),
            "delta_ece": baseline.get("ece", 0)
            - test_metrics.get("ece", 0),  # Lower is better
        }
        comparison.append(delta)

    return pd.DataFrame(comparison)
