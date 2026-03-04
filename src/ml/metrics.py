"""
Common regression metrics for ML model evaluation.
"""

from typing import Dict

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


__all__ = ["regression_metrics"]

