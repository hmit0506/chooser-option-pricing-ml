"""
Week 8 pricing tool services:
- Dual pricing (Rubinstein BSM + best ML model)
- Error margin estimation from Week 6 backtest residuals
- Dashboard data assembly (trend, metrics, sensitivity)
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml

from src.ml.datasets import build_pricing_dataset, load_base_frame
from src.models import rubinstein_chooser

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "model_params.yaml"
WEEK6_RESULTS_PATH = PROJECT_ROOT / "data" / "reports" / "week6" / "week6_results.json"
WEEK6_COMPARISON_PATH = PROJECT_ROOT / "data" / "reports" / "week6" / "model_comparison.csv"
WEEK7_SENS_PATH = PROJECT_ROOT / "data" / "reports" / "week7"


@dataclass
class ToolContext:
    config: Dict[str, Any]
    frame: pd.DataFrame
    pricing_df: pd.DataFrame
    feature_cols: List[str]
    ml_model: Any
    residual_std: float
    residual_mae: float


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _best_ml_model_path_from_week6() -> Path:
    if WEEK6_RESULTS_PATH.exists():
        with open(WEEK6_RESULTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        rel = data.get("artifacts", {}).get("best_pricing_model")
        if rel:
            return PROJECT_ROOT / rel
    fallback = PROJECT_ROOT / "models" / "week6" / "best_pricing_model_mlp.pkl"
    return fallback


def _build_pricing_frame(frame: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    k = float(cfg["k"])
    r_default = float(cfg["r"])
    q = float(cfg["q"])
    t1 = float(cfg["t1"])
    t2 = float(cfg["t2"])
    t1_days = int(round(t1 * 252))
    t2_days = int(round(t2 * 252))

    X_df, y, dates, bsm = build_pricing_dataset(
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
    out = X_df.copy()
    out.index = pd.to_datetime(dates)
    out["target"] = y
    out["bsm_price"] = bsm
    return out


def load_tool_context() -> ToolContext:
    cfg = _load_yaml(CONFIG_PATH)["model"]
    frame = load_base_frame()
    pricing_df = _build_pricing_frame(frame, cfg)
    feature_cols = [c for c in pricing_df.columns if c not in {"target"}]

    ml_model_path = _best_ml_model_path_from_week6()
    ml_model = _load_pickle(ml_model_path)

    y_true = pricing_df["target"].values
    y_pred = np.asarray(ml_model.predict(pricing_df[feature_cols].values)).reshape(-1)
    residual = y_true - y_pred
    residual_std = float(np.std(residual))
    residual_mae = float(np.mean(np.abs(residual)))

    return ToolContext(
        config=cfg,
        frame=frame,
        pricing_df=pricing_df,
        feature_cols=feature_cols,
        ml_model=ml_model,
        residual_std=residual_std,
        residual_mae=residual_mae,
    )


def _latest_feature_row(ctx: ToolContext) -> pd.Series:
    return ctx.pricing_df.iloc[-1].copy()


def _build_feature_vector(
    ctx: ToolContext,
    s: float,
    k: float,
    r: float,
    q: float,
    sigma: float,
    t1: float,
    t2: float,
    vix: float,
    sentiment: float,
) -> pd.DataFrame:
    row = _latest_feature_row(ctx)
    row["s_t"] = float(s)
    row["moneyness_t"] = float(s / k) if k > 0 else np.nan
    row["r_t"] = float(r)
    row["sigma_t"] = float(max(sigma, 1e-8))
    row["vix_t"] = float(vix)
    row["sentiment_proxy"] = float(sentiment)
    row["bsm_price"] = float(rubinstein_chooser(s, k, r, q, max(sigma, 1e-8), t1, t2))
    return pd.DataFrame([row[ctx.feature_cols].values], columns=ctx.feature_cols)


def dual_price(
    ctx: ToolContext,
    s: float,
    k: float,
    r: float,
    q: float,
    sigma: float,
    t1: float,
    t2: float,
    vix: float,
    sentiment: float,
) -> Dict[str, float]:
    bsm = float(rubinstein_chooser(s, k, r, q, max(sigma, 1e-8), t1, t2))
    X = _build_feature_vector(ctx, s, k, r, q, sigma, t1, t2, vix, sentiment)
    ml = float(np.asarray(ctx.ml_model.predict(X.values)).reshape(-1)[0])

    # Approximate error margins from historical residual distribution.
    margin_68 = float(ctx.residual_std)
    margin_95 = float(1.96 * ctx.residual_std)
    return {
        "bsm_price": bsm,
        "ml_price": ml,
        "delta_ml_minus_bsm": ml - bsm,
        "error_margin_68": margin_68,
        "error_margin_95": margin_95,
        "historical_residual_mae": float(ctx.residual_mae),
    }


def dashboard_series(ctx: ToolContext, n_points: int = 200) -> pd.DataFrame:
    df = ctx.pricing_df.copy().tail(n_points)
    X = df[ctx.feature_cols].values
    df["ml_price"] = np.asarray(ctx.ml_model.predict(X)).reshape(-1)
    df["ml_abs_err"] = np.abs(df["target"] - df["ml_price"])
    df["bsm_abs_err"] = np.abs(df["target"] - df["bsm_price"])
    return df[["target", "bsm_price", "ml_price", "ml_abs_err", "bsm_abs_err"]]


def performance_metrics() -> pd.DataFrame:
    if WEEK6_COMPARISON_PATH.exists():
        return pd.read_csv(WEEK6_COMPARISON_PATH)
    return pd.DataFrame(columns=["model", "mae", "rmse", "r2"])


def sensitivity_tables() -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    p1 = WEEK7_SENS_PATH / "shap_by_maturity_bucket.csv"
    p2 = WEEK7_SENS_PATH / "shap_by_moneyness_bucket.csv"
    p3 = WEEK7_SENS_PATH / "historical_event_calibration.csv"
    if p1.exists():
        out["by_maturity"] = pd.read_csv(p1)
    if p2.exists():
        out["by_moneyness"] = pd.read_csv(p2)
    if p3.exists():
        out["historical_event_calibration"] = pd.read_csv(p3)
    return out


__all__ = [
    "ToolContext",
    "load_tool_context",
    "dual_price",
    "dashboard_series",
    "performance_metrics",
    "sensitivity_tables",
]
