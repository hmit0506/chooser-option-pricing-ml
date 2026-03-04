"""
Common regression metrics for ML model evaluation.
"""

from typing import Dict, Optional

import numpy as np


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    """
    Compute MAE, RMSE and R^2 between arrays of true and predicted values.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true

    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err**2))

    # R^2
    ss_res = np.sum(err**2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}


def relative_improvement(
    ml_metric: float,
    baseline_metric: float,
) -> float:
    """
    Percentage improvement over baseline for error metrics.

    Positive values mean ML is better (lower error).
    """
    if baseline_metric == 0:
        return np.nan
    return (baseline_metric - ml_metric) / baseline_metric * 100.0


def benchmark_against_baseline(
    y_true,
    y_pred,
    y_baseline: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute model metrics and optional improvements vs baseline predictions.

    Args:
        y_true: Ground-truth labels.
        y_pred: ML predictions.
        y_baseline: Optional baseline predictions (e.g., BSM prices).

    Returns:
        Dictionary with MAE/RMSE/R2 and optional improvement_% fields.
    """
    out = regression_metrics(y_true, y_pred)

    if y_baseline is not None:
        base = regression_metrics(y_true, y_baseline)
        out["baseline_mae"] = base["mae"]
        out["baseline_rmse"] = base["rmse"]
        out["mae_improvement_pct"] = relative_improvement(out["mae"], base["mae"])
        out["rmse_improvement_pct"] = relative_improvement(out["rmse"], base["rmse"])

    return out


__all__ = ["regression_metrics", "relative_improvement", "benchmark_against_baseline"]

