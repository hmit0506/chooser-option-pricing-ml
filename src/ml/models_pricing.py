"""
End-to-end pricing models for Approach 2.
"""

from typing import Dict, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import benchmark_against_baseline, regression_metrics


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
    # Scale-sensitive models are wrapped in a pipeline.
    base = Ridge(alpha=alpha) if use_ridge else LinearRegression()
    model = Pipeline([("scaler", StandardScaler()), ("model", base)])
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
    base = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        max_iter=max_iter,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
    )
    model = Pipeline([("scaler", StandardScaler()), ("model", base)])
    model.fit(X_train, y_train)
    y_pred_val = model.predict(X_val)
    metrics = regression_metrics(y_val, y_pred_val)
    return model, metrics


def evaluate_pricing_model(
    model,
    X: np.ndarray,
    y_true: np.ndarray,
    y_baseline: np.ndarray | None = None,
) -> Dict[str, float]:
    """
    Evaluate a pricing model and optionally compare against baseline predictions.
    """
    y_pred = np.asarray(model.predict(X)).reshape(-1)
    return benchmark_against_baseline(y_true, y_pred, y_baseline=y_baseline)


def train_and_evaluate_pricing_model(
    trainer_fn,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_baseline_test: np.ndarray | None = None,
    **trainer_kwargs,
) -> Tuple[object, Dict[str, Dict[str, float]]]:
    """
    Common wrapper to train a pricing model and return val/test metrics.

    Args:
        trainer_fn: One of train_linear_pricing / train_gbdt_pricing / train_mlp_pricing.
    """
    model, val_metrics = trainer_fn(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        **trainer_kwargs,
    )
    test_metrics = evaluate_pricing_model(
        model,
        X_test,
        y_test,
        y_baseline=y_baseline_test,
    )
    return model, {"val": val_metrics, "test": test_metrics}


__all__ = [
    "train_linear_pricing",
    "train_gbdt_pricing",
    "train_mlp_pricing",
    "evaluate_pricing_model",
    "train_and_evaluate_pricing_model",
]

