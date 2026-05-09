from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


@dataclass(frozen=True)
class XGBoostConfig:
    learning_rate: float = 0.05
    max_depth: int = 6
    n_estimators: int = 500
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 5
    gamma: float = 0.1
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 50
    eval_metric: str = "logloss"
    tree_method: str = "hist"
    random_state: int = 42

    def to_xgb_params(self) -> dict:
        return {
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "n_estimators": self.n_estimators,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "gamma": self.gamma,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "eval_metric": self.eval_metric,
            "tree_method": self.tree_method,
            "random_state": self.random_state,
        }


@dataclass(frozen=True)
class CalibrationConfig:
    method: str = "isotonic"
    cv_folds: int = 5


@dataclass(frozen=True)
class PathConfig:
    model_dir: Path = field(default_factory=lambda: RESULTS_DIR / "models")
    shap_dir: Path = field(default_factory=lambda: RESULTS_DIR / "shap")
    plot_dir: Path = field(default_factory=lambda: RESULTS_DIR / "plots")
    baseline_model_path: Path = field(
        default_factory=lambda: RESULTS_DIR / "models" / "xgb_baseline.json"
    )
