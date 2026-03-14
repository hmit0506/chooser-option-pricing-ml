"""
Week 6 pipeline:
1) Hyperparameter optimization with time-series CV
2) Final training and test evaluation
3) ML vs BSM benchmark comparison
4) SHAP + LIME interpretability artifacts
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.datasets import (
    build_pricing_dataset,
    build_volatility_dataset,
    load_base_frame,
    time_series_split,
)
from src.ml.metrics import benchmark_against_baseline, regression_metrics
from src.models import rubinstein_chooser

try:
    import shap
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("shap is required for Week 6 interpretability outputs") from exc


CONFIG_PATH = PROJECT_ROOT / "config" / "model_params.yaml"
MODELS_DIR = PROJECT_ROOT / "models" / "week6"
REPORT_DIR = PROJECT_ROOT / "data" / "reports" / "week6"
PLOTS_DIR = REPORT_DIR / "plots"
SUMMARY_JSON = REPORT_DIR / "week6_results.json"


@dataclass
class SplitData:
    X_train: np.ndarray
    y_train: np.ndarray
    d_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    d_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    d_test: np.ndarray


def ensure_dirs() -> None:
    for path in [MODELS_DIR, REPORT_DIR, PLOTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_params() -> Dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def split_with_val_in_train(X: pd.DataFrame, y: np.ndarray, d: np.ndarray) -> SplitData:
    (X_train, y_train, d_train), (X_val, y_val, d_val), (X_test, y_test, d_test) = time_series_split(
        X, y, d, train_frac=0.7, val_frac=0.15
    )
    return SplitData(
        X_train=X_train,
        y_train=y_train,
        d_train=d_train,
        X_val=X_val,
        y_val=y_val,
        d_val=d_val,
        X_test=X_test,
        y_test=y_test,
        d_test=d_test,
    )


def make_ts_cv(n_splits: int = 4) -> TimeSeriesSplit:
    return TimeSeriesSplit(n_splits=n_splits)


def tune_rf_vol(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> RandomForestRegressor:
    cv = make_ts_cv()
    scoring = make_scorer(mean_absolute_error, greater_is_better=False)
    grid = GridSearchCV(
        estimator=RandomForestRegressor(random_state=random_state, n_jobs=-1),
        param_grid={
            "n_estimators": [200, 400],
            "max_depth": [4, 6, None],
            "min_samples_leaf": [1, 3, 8],
        },
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X, y)
    return grid.best_estimator_


def tune_xgb_vol(X: np.ndarray, y: np.ndarray, random_state: int = 42):
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("xgboost is not installed in this environment") from exc

    cv = make_ts_cv()
    scoring = make_scorer(mean_absolute_error, greater_is_better=False)
    search = RandomizedSearchCV(
        estimator=XGBRegressor(
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=1,
            tree_method="hist",
        ),
        param_distributions={
            "n_estimators": [250, 400, 600],
            "max_depth": [3, 4, 5],
            "learning_rate": [0.02, 0.05, 0.08],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.7, 0.85, 1.0],
        },
        n_iter=12,
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        n_jobs=1,
        refit=True,
    )
    search.fit(X, y)
    return search.best_estimator_


def tune_pricing_models(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> Dict[str, Any]:
    cv = make_ts_cv()
    scoring = make_scorer(mean_absolute_error, greater_is_better=False)

    ridge_pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
    ridge_grid = GridSearchCV(
        estimator=ridge_pipe,
        param_grid={"model__alpha": [0.1, 1.0, 5.0, 10.0]},
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    ridge_grid.fit(X, y)

    gbdt_search = RandomizedSearchCV(
        estimator=GradientBoostingRegressor(random_state=random_state),
        param_distributions={
            "n_estimators": [200, 350, 500],
            "max_depth": [2, 3, 4],
            "learning_rate": [0.02, 0.05, 0.08],
            "subsample": [0.7, 0.85, 1.0],
            "min_samples_leaf": [1, 3, 8],
        },
        n_iter=12,
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        refit=True,
    )
    gbdt_search.fit(X, y)

    mlp_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                MLPRegressor(
                    random_state=random_state,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=20,
                    max_iter=800,
                ),
            ),
        ]
    )
    mlp_search = RandomizedSearchCV(
        estimator=mlp_pipe,
        param_distributions={
            "model__hidden_layer_sizes": [(64, 32), (128, 64), (128, 64, 32)],
            "model__alpha": [1e-5, 1e-4, 1e-3],
            "model__learning_rate_init": [1e-4, 5e-4, 1e-3],
        },
        n_iter=9,
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        refit=True,
    )
    mlp_search.fit(X, y)

    return {
        "ridge": ridge_grid.best_estimator_,
        "gbdt": gbdt_search.best_estimator_,
        "mlp": mlp_search.best_estimator_,
    }


def price_with_predicted_vol(
    pricing_X: pd.DataFrame,
    vol_pred_by_date: pd.Series,
    k: float,
    q: float,
    t1: float,
    t2: float,
) -> Tuple[np.ndarray, np.ndarray]:
    aligned = pricing_X.copy()
    aligned["sigma_pred"] = vol_pred_by_date.reindex(aligned.index)
    aligned = aligned.dropna(subset=["sigma_pred"])

    preds = []
    out_dates = []
    for dt, row in aligned.iterrows():
        sigma_val = max(float(row["sigma_pred"]), 1e-6)
        preds.append(
            rubinstein_chooser(
                s=float(row["s_t"]),
                k=k,
                r=float(row["r_t"]),
                q=q,
                sigma=sigma_val,
                t1=t1,
                t2=t2,
            )
        )
        out_dates.append(dt)
    return np.asarray(preds), np.asarray(out_dates)


def save_pickle(obj: Any, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def make_comparison_plot(df: pd.DataFrame, path: Path) -> None:
    labels = df["model"].tolist()
    mae = df["mae"].values
    rmse = df["rmse"].values

    x = np.arange(len(labels))
    width = 0.38

    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, mae, width=width, label="MAE")
    plt.bar(x + width / 2, rmse, width=width, label="RMSE")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("Error")
    plt.title("Week 6 Test Error Comparison (lower is better)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def build_shap_outputs(
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
) -> Dict[str, str]:
    # Use TreeExplainer directly for GradientBoostingRegressor.
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    summary_png = PLOTS_DIR / "shap_summary.png"
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(summary_png, dpi=160, bbox_inches="tight")
    plt.close()

    bar_png = PLOTS_DIR / "shap_bar.png"
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(bar_png, dpi=160, bbox_inches="tight")
    plt.close()

    return {
        "shap_summary": str(summary_png.relative_to(PROJECT_ROOT)),
        "shap_bar": str(bar_png.relative_to(PROJECT_ROOT)),
    }


def build_lime_output(model: Any, X_train: np.ndarray, X_test: np.ndarray, feature_names: list[str]) -> str:
    explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        mode="regression",
    )
    exp = explainer.explain_instance(X_test[0], model.predict, num_features=min(8, len(feature_names)))
    lime_html = REPORT_DIR / "lime_explanation_sample0.html"
    exp.save_to_file(str(lime_html))
    return str(lime_html.relative_to(PROJECT_ROOT))


def main() -> None:
    ensure_dirs()
    params = load_params()
    model_cfg = params["model"]

    k = float(model_cfg["k"])
    r_default = float(model_cfg["r"])
    q = float(model_cfg["q"])
    t1 = float(model_cfg["t1"])
    t2 = float(model_cfg["t2"])
    t1_days = int(round(t1 * 252))
    t2_days = int(round(t2 * 252))

    frame = load_base_frame()

    # ------------------------------------------------------------------
    # Approach 1: Volatility prediction -> BSM pricing
    # ------------------------------------------------------------------
    X_vol_df, y_vol, d_vol = build_volatility_dataset(frame=frame, horizon_days=t2_days)
    vol_split = split_with_val_in_train(X_vol_df, y_vol, d_vol)
    X_vol_trainval = np.vstack([vol_split.X_train, vol_split.X_val])
    y_vol_trainval = np.concatenate([vol_split.y_train, vol_split.y_val])

    rf_vol = tune_rf_vol(X_vol_trainval, y_vol_trainval)
    xgb_vol = tune_xgb_vol(X_vol_trainval, y_vol_trainval)

    rf_vol_test_metrics = regression_metrics(vol_split.y_test, rf_vol.predict(vol_split.X_test))
    xgb_vol_test_metrics = regression_metrics(vol_split.y_test, xgb_vol.predict(vol_split.X_test))

    vol_test_dates = pd.to_datetime(vol_split.d_test)
    rf_vol_pred_series = pd.Series(rf_vol.predict(vol_split.X_test), index=vol_test_dates)
    xgb_vol_pred_series = pd.Series(xgb_vol.predict(vol_split.X_test), index=vol_test_dates)

    X_pricing_df, y_price, d_price, bsm_price = build_pricing_dataset(
        frame=frame,
        k=k,
        r_default=r_default,
        q=q,
        t1_years=t1,
        t2_years=t2,
        t1_days=t1_days,
        t2_days=t2_days,
        target_mode="direct",
        include_bsm_feature=False,
    )
    pricing_df = X_pricing_df.copy()
    pricing_df["target"] = y_price
    pricing_df["bsm_price"] = bsm_price
    pricing_df.index = pd.to_datetime(pricing_df.index)

    rf_price_pred, rf_dates = price_with_predicted_vol(pricing_df, rf_vol_pred_series, k, q, t1, t2)
    xgb_price_pred, xgb_dates = price_with_predicted_vol(pricing_df, xgb_vol_pred_series, k, q, t1, t2)

    rf_eval = pricing_df.reindex(pd.to_datetime(rf_dates)).dropna(subset=["target", "bsm_price"])
    xgb_eval = pricing_df.reindex(pd.to_datetime(xgb_dates)).dropna(subset=["target", "bsm_price"])

    rf_approach1_metrics = benchmark_against_baseline(
        rf_eval["target"].values,
        rf_price_pred[: len(rf_eval)],
        y_baseline=rf_eval["bsm_price"].values,
    )
    xgb_approach1_metrics = benchmark_against_baseline(
        xgb_eval["target"].values,
        xgb_price_pred[: len(xgb_eval)],
        y_baseline=xgb_eval["bsm_price"].values,
    )

    # ------------------------------------------------------------------
    # Approach 2: End-to-end pricing
    # ------------------------------------------------------------------
    X_price_df, y_price_full, d_price_full, bsm_price_full = build_pricing_dataset(
        frame=frame,
        k=k,
        r_default=r_default,
        q=q,
        t1_years=t1,
        t2_years=t2,
        t1_days=t1_days,
        t2_days=t2_days,
        target_mode="direct",
        include_bsm_feature=True,
    )
    price_split = split_with_val_in_train(X_price_df, y_price_full, d_price_full)
    n_trainval = len(price_split.y_train) + len(price_split.y_val)
    y_base_test = bsm_price_full[n_trainval:]

    X_price_trainval = np.vstack([price_split.X_train, price_split.X_val])
    y_price_trainval = np.concatenate([price_split.y_train, price_split.y_val])
    tuned_pricing = tune_pricing_models(X_price_trainval, y_price_trainval)

    approach2_metrics: Dict[str, Dict[str, float]] = {}
    for name, model in tuned_pricing.items():
        y_pred_test = model.predict(price_split.X_test)
        approach2_metrics[name] = benchmark_against_baseline(
            price_split.y_test,
            y_pred_test,
            y_baseline=y_base_test,
        )

    # Primary model choices
    best_vol_name = min(
        [("rf_vol", rf_approach1_metrics), ("xgb_vol", xgb_approach1_metrics)],
        key=lambda x: x[1]["mae"],
    )[0]
    best_pricing_name = min(approach2_metrics.items(), key=lambda x: x[1]["mae"])[0]

    best_vol_model = rf_vol if best_vol_name == "rf_vol" else xgb_vol
    best_pricing_model = tuned_pricing[best_pricing_name]

    save_pickle(best_vol_model, MODELS_DIR / f"best_vol_model_{best_vol_name}.pkl")
    save_pickle(best_pricing_model, MODELS_DIR / f"best_pricing_model_{best_pricing_name}.pkl")

    # Interpretability on best end-to-end model.
    if best_pricing_name != "gbdt":
        # SHAP TreeExplainer is strongest for tree model; still produce consistent outputs by
        # using the best available tree model for interpretation artifacts.
        shap_model = tuned_pricing["gbdt"]
        shap_model_name = "gbdt"
    else:
        shap_model = best_pricing_model
        shap_model_name = best_pricing_name

    feature_names = list(X_price_df.columns)
    shap_paths = build_shap_outputs(
        model=shap_model,
        X_train=X_price_trainval,
        X_test=price_split.X_test,
        feature_names=feature_names,
    )
    lime_path = build_lime_output(
        model=shap_model,
        X_train=X_price_trainval,
        X_test=price_split.X_test,
        feature_names=feature_names,
    )

    comp_rows = [
        {"model": "bsm_baseline", **regression_metrics(price_split.y_test, y_base_test)},
        {"model": "approach1_rf_vol_bsm", **rf_approach1_metrics},
        {"model": "approach1_xgb_vol_bsm", **xgb_approach1_metrics},
        {"model": "approach2_ridge", **approach2_metrics["ridge"]},
        {"model": "approach2_gbdt", **approach2_metrics["gbdt"]},
        {"model": "approach2_mlp", **approach2_metrics["mlp"]},
    ]
    comp_df = pd.DataFrame(comp_rows)
    comp_csv_path = REPORT_DIR / "model_comparison.csv"
    comp_df.to_csv(comp_csv_path, index=False)
    make_comparison_plot(comp_df[["model", "mae", "rmse"]], PLOTS_DIR / "model_error_comparison.png")

    results = {
        "config": {"k": k, "q": q, "t1": t1, "t2": t2, "t1_days": t1_days, "t2_days": t2_days},
        "sample_sizes": {
            "volatility_train": int(len(vol_split.y_train)),
            "volatility_val": int(len(vol_split.y_val)),
            "volatility_test": int(len(vol_split.y_test)),
            "pricing_train": int(len(price_split.y_train)),
            "pricing_val": int(len(price_split.y_val)),
            "pricing_test": int(len(price_split.y_test)),
        },
        "approach1": {
            "vol_model_test_metrics": {
                "rf_vol": rf_vol_test_metrics,
                "xgb_vol": xgb_vol_test_metrics,
            },
            "pricing_metrics_with_predicted_vol": {
                "rf_vol_bsm": rf_approach1_metrics,
                "xgb_vol_bsm": xgb_approach1_metrics,
            },
            "best_vol_model": best_vol_name,
        },
        "approach2": {
            "pricing_test_metrics": approach2_metrics,
            "best_pricing_model": best_pricing_name,
        },
        "artifacts": {
            "best_vol_model": str((MODELS_DIR / f"best_vol_model_{best_vol_name}.pkl").relative_to(PROJECT_ROOT)),
            "best_pricing_model": str(
                (MODELS_DIR / f"best_pricing_model_{best_pricing_name}.pkl").relative_to(PROJECT_ROOT)
            ),
            "comparison_csv": str(comp_csv_path.relative_to(PROJECT_ROOT)),
            "comparison_plot": str((PLOTS_DIR / "model_error_comparison.png").relative_to(PROJECT_ROOT)),
            "shap_model_used": shap_model_name,
            **shap_paths,
            "lime_html": lime_path,
        },
    }
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("=" * 70)
    print("WEEK 6 TRAINING COMPLETED")
    print("=" * 70)
    print(f"Best vol model (Approach 1): {best_vol_name}")
    print(f"Best pricing model (Approach 2): {best_pricing_name}")
    print(f"Summary JSON: {SUMMARY_JSON}")
    print(f"Comparison CSV: {comp_csv_path}")
    print(f"SHAP summary: {PLOTS_DIR / 'shap_summary.png'}")
    print(f"LIME HTML: {REPORT_DIR / 'lime_explanation_sample0.html'}")


if __name__ == "__main__":
    main()
