"""XGBoost baseline for 30-day readmission prediction."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

try:
    import xgboost as xgb
except ImportError:
    sys.exit("xgboost is required: pip install xgboost")

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    sys.exit("scikit-learn is required: pip install scikit-learn")

try:
    import shap
except ImportError:
    shap = None

from configs.model_config import PathConfig, XGBoostConfig
from src.data.eicu_loader import EICULoader
from src.eval.eval_utils import compute_all_metrics

logger = logging.getLogger(__name__)


def _scale_pos_weight(y: NDArray) -> float:
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    if n_pos == 0:
        return 1.0
    return float(n_neg / n_pos)


def train_xgb_baseline(
    xgb_cfg: XGBoostConfig | None = None,
    path_cfg: PathConfig | None = None,
) -> dict:
    xgb_cfg = xgb_cfg or XGBoostConfig()
    path_cfg = path_cfg or PathConfig()

    loader = EICULoader()
    features, labels = loader.build_features()
    y = labels["readmission_30d"].values

    X_train, y_train, X_test, y_test = loader.time_split(features, labels)
    y_train = y_train["readmission_30d"].values
    y_test = y_test["readmission_30d"].values

    # Carve a validation set from the training split (last 20% by index order)
    split_idx = int(len(X_train) * 0.8)
    X_val, y_val = X_train.iloc[split_idx:], y_train[split_idx:]
    X_train, y_train = X_train.iloc[:split_idx], y_train[:split_idx]

    spw = _scale_pos_weight(y_train)
    logger.info("scale_pos_weight=%.2f", spw)

    params = xgb_cfg.to_xgb_params()
    params["scale_pos_weight"] = spw

    model = xgb.XGBClassifier(
        **params,
        n_estimators=xgb_cfg.n_estimators,
        early_stopping_rounds=xgb_cfg.early_stopping_rounds,
        use_label_encoder=False,
    )
    # n_estimators is already in params, remove duplicate before fit
    model.set_params(n_estimators=xgb_cfg.n_estimators)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    # --- Evaluate ---
    y_prob_test = model.predict_proba(X_test)[:, 1]
    metrics = compute_all_metrics(y_test, y_prob_test)

    logger.info("Test metrics:")
    for k, v in metrics.items():
        logger.info("  %s: %.4f", k, v)

    # --- SHAP feature importance ---
    shap_values = None
    if shap is not None:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[::-1][:20]
        feature_names = list(X_test.columns)
        top_features = [
            {"feature": feature_names[i], "mean_abs_shap": float(mean_abs_shap[i])}
            for i in top_indices
        ]
        logger.info("Top 20 features by SHAP:")
        for entry in top_features:
            logger.info("  %s: %.4f", entry["feature"], entry["mean_abs_shap"])
    else:
        top_features = []
        logger.warning("shap not installed — skipping feature importance")

    # --- Save model ---
    path_cfg.model_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path_cfg.baseline_model_path))
    logger.info("Model saved to %s", path_cfg.baseline_model_path)

    # --- Save SHAP summary ---
    if top_features:
        shap_path = path_cfg.shap_dir / "xgb_baseline_shap.json"
        path_cfg.shap_dir.mkdir(parents=True, exist_ok=True)
        with open(shap_path, "w") as f:
            json.dump(top_features, f, indent=2)

    result = {
        "metrics": metrics,
        "top_features": top_features,
        "model_path": str(path_cfg.baseline_model_path),
        "train_size": len(y_train),
        "val_size": len(y_val),
        "test_size": len(y_test),
        "scale_pos_weight": spw,
        "best_iteration": model.best_iteration if hasattr(model, "best_iteration") else None,
    }
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    result = train_xgb_baseline()
    print(json.dumps(result["metrics"], indent=2))
