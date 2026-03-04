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

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover - optional dependency
    tf = None  # type: ignore


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


def train_lstm_vol(
    X_train_seq: np.ndarray,
    y_train: np.ndarray,
    X_val_seq: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    random_state: int = 42,
) -> Tuple[Any, Dict[str, float]]:
    """
    Train a simple LSTM model for volatility forecasting.

    Expected shapes:
      X_train_seq: (n_samples, lookback_days, n_features)
      y_train: (n_samples,)
    """
    if tf is None:
        raise RuntimeError("tensorflow is not installed in this environment")

    tf.keras.utils.set_random_seed(random_state)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
            tf.keras.layers.LSTM(32, return_sequences=False),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )

    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )
    model.fit(
        X_train_seq,
        y_train,
        validation_data=(X_val_seq, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[es],
    )

    y_pred_val = model.predict(X_val_seq, verbose=0).reshape(-1)
    metrics = regression_metrics(y_val, y_pred_val)
    return model, metrics


def predict_vol(model: Any, X: np.ndarray) -> np.ndarray:
    """
    Unified prediction helper for sklearn and keras models.
    """
    if hasattr(model, "predict"):
        y_hat = model.predict(X)
        y_hat = np.asarray(y_hat).reshape(-1)
        return y_hat
    raise TypeError("model does not implement predict()")


def evaluate_vol_model(
    model: Any,
    X: np.ndarray,
    y_true: np.ndarray,
) -> Dict[str, float]:
    """
    Unified volatility model evaluation.
    """
    y_pred = predict_vol(model, X)
    return regression_metrics(y_true, y_pred)


def train_and_evaluate_vol_model(
    trainer_fn,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    **trainer_kwargs: Any,
) -> Tuple[Any, Dict[str, Dict[str, float]]]:
    """
    Common wrapper to train a vol model and return val/test metrics.

    Args:
        trainer_fn: One of train_rf_vol / train_xgb_vol / train_lstm_vol.
    """
    model, val_metrics = trainer_fn(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        **trainer_kwargs,
    )
    test_metrics = evaluate_vol_model(model, X_test, y_test)
    return model, {"val": val_metrics, "test": test_metrics}


__all__ = [
    "train_rf_vol",
    "train_xgb_vol",
    "train_lstm_vol",
    "predict_vol",
    "evaluate_vol_model",
    "train_and_evaluate_vol_model",
]

