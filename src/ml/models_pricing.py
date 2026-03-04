"""
End-to-end pricing models for Approach 2.
"""

from typing import Dict, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

from .metrics import regression_metrics


def train_linear_pricing(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    use_ridge: bool = False,
    alpha: float = 1.0,
) -> Tuple[object, Dict[str, float]]:
    """
    Train a linear (or Ridge) regression model as a pricing baseline.
    """
    model = Ridge(alpha=alpha) if use_ridge else LinearRegression()
    model.fit(X_train, y_train)
    y_pred_val = model.predict(X_val)
    metrics = regression_metrics(y_val, y_pred_val)
    return model, metrics


def train_gbdt_pricing(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_estimators: int = 300,
    max_depth: int = 3,
    learning_rate: float = 0.05,
) -> Tuple[GradientBoostingRegressor, Dict[str, float]]:
    """
    Train a Gradient Boosted Trees regressor for pricing.
    """
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
    )
    model.fit(X_train, y_train)
    y_pred_val = model.predict(X_val)
    metrics = regression_metrics(y_val, y_pred_val)
    return model, metrics


def train_mlp_pricing(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden_layer_sizes=(64, 32),
    max_iter: int = 500,
    random_state: int = 42,
) -> Tuple[MLPRegressor, Dict[str, float]]:
    """
    Train a small MLP regressor for pricing.
    """
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        max_iter=max_iter,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    y_pred_val = model.predict(X_val)
    metrics = regression_metrics(y_val, y_pred_val)
    return model, metrics


__all__ = ["train_linear_pricing", "train_gbdt_pricing", "train_mlp_pricing"]

