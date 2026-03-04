"""
Volatility forecasting models for Approach 1.
"""

from typing import Any, Dict, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover - optional dependency
    XGBRegressor = None  # type: ignore

from .metrics import regression_metrics


def train_rf_vol(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_estimators: int = 200,
    max_depth: int | None = None,
    random_state: int = 42,
) -> Tuple[RandomForestRegressor, Dict[str, float]]:
    """
    Train a Random Forest regressor for volatility forecasting.
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred_val = model.predict(X_val)
    metrics = regression_metrics(y_val, y_pred_val)
    return model, metrics


def train_xgb_vol(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    random_state: int = 42,
    **kwargs: Any,
) -> Tuple[Any, Dict[str, float]]:
    """
    Train an XGBoost regressor for volatility forecasting.

    If xgboost is not installed, raises a RuntimeError.
    """
    if XGBRegressor is None:
        raise RuntimeError("xgboost is not installed in this environment")

    params = {
        "n_estimators": 400,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": random_state,
        "objective": "reg:squarederror",
        **kwargs,
    }
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    y_pred_val = model.predict(X_val)
    metrics = regression_metrics(y_val, y_pred_val)
    return model, metrics


__all__ = ["train_rf_vol", "train_xgb_vol"]

