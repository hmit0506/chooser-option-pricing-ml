"""
Week 7 extended sensitivity analysis:
1) SHAP mean |impact| for VIX and sentiment (and other pricing features) on a tree model
2) Extreme BSM chooser scenarios: +50% vol, +200 bps rates, combined
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.ensemble import GradientBoostingRegressor

from src.ml.datasets import build_pricing_dataset, load_base_frame, time_series_split
from src.models import rubinstein_chooser

try:
    import shap
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("shap is required for Week 7 sensitivity analysis") from exc

CONFIG_PATH = PROJECT_ROOT / "config" / "model_params.yaml"
REPORT_DIR = PROJECT_ROOT / "data" / "reports" / "week7"
PLOTS_DIR = REPORT_DIR / "plots"


def ensure_dirs() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_params() -> Dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_gbdt_for_shap(
    X_trainval: np.ndarray,
    y_trainval: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42,
) -> Tuple[GradientBoostingRegressor, Dict[str, float]]:
    """Small GBDT aligned with Week 6 interpretability (tree SHAP)."""
    model = GradientBoostingRegressor(
        random_state=random_state,
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
    )
    model.fit(X_trainval, y_trainval)
    pred = model.predict(X_test)
    mae = float(np.mean(np.abs(pred - y_test)))
    rmse = float(np.sqrt(np.mean((pred - y_test) ** 2)))
    return model, {"mae": mae, "rmse": rmse}


def extreme_scenario_table(
    s: float,
    k: float,
    r: float,
    q: float,
    sigma: float,
    t1: float,
    t2: float,
) -> pd.DataFrame:
    """
    Baseline Rubinstein chooser + stresses:
    -     Vol +50%: sigma * 1.5
    - Rates +200 bps: r + 0.02 (absolute)
    - Combined: both shocks
    """
    base = rubinstein_chooser(s, k, r, q, sigma, t1, t2)
    vol_shock = rubinstein_chooser(s, k, r, q, sigma * 1.5, t1, t2)
    rate_shock = rubinstein_chooser(s, k, r + 0.02, q, sigma, t1, t2)
    both = rubinstein_chooser(s, k, r + 0.02, q, sigma * 1.5, t1, t2)

    def pct_change(x: float) -> float:
        if base == 0:
            return float("nan")
        return (x - base) / base * 100.0

    rows = [
        {
            "scenario": "baseline",
            "chooser_price": base,
            "pct_vs_baseline": 0.0,
            "description": "Rubinstein closed-form, config / last-row inputs",
        },
        {
            "scenario": "vol_spike_50pct",
            "chooser_price": vol_shock,
            "pct_vs_baseline": pct_change(vol_shock),
            "description": "sigma multiplied by 1.5 (+50% volatility)",
        },
        {
            "scenario": "rate_hike_200bps",
            "chooser_price": rate_shock,
            "pct_vs_baseline": pct_change(rate_shock),
            "description": "risk-free rate +0.02 (+200 bps absolute)",
        },
        {
            "scenario": "vol_spike_and_rate_hike",
            "chooser_price": both,
            "pct_vs_baseline": pct_change(both),
            "description": "both shocks applied simultaneously",
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()
    params = load_params()
    m = params["model"]
    k = float(m["k"])
    r_default = float(m["r"])
    q = float(m["q"])
    t1 = float(m["t1"])
    t2 = float(m["t2"])
    t1_days = int(round(t1 * 252))
    t2_days = int(round(t2 * 252))

    frame = load_base_frame()
    X_df, y, d, _bsm = build_pricing_dataset(
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
    feature_names = list(X_df.columns)

    (X_train, y_train, _dt), (X_val, y_val, _dv), (X_test, y_test, _dte) = time_series_split(
        X_df, y, d, train_frac=0.7, val_frac=0.15
    )
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])

    # SHAP: explain test set (subsample for speed)
    rng = np.random.default_rng(42)
    n_explain = min(800, len(X_test))
    idx = rng.choice(len(X_test), size=n_explain, replace=False)
    X_exp = X_test[idx]

    model, gbdt_metrics = train_gbdt_for_shap(X_trainval, y_trainval, X_test, y_test)

    n_bg = min(400, len(X_trainval))
    bg_idx = rng.choice(len(X_trainval), size=n_bg, replace=False)
    X_bg = X_trainval[bg_idx]

    explainer = shap.TreeExplainer(model, data=X_bg)
    shap_values = explainer.shap_values(X_exp)

    impact = pd.DataFrame(
        {"feature": feature_names, "mean_abs_shap": np.mean(np.abs(shap_values), axis=0)}
    ).sort_values("mean_abs_shap", ascending=False)
    impact_path = REPORT_DIR / "shap_mean_abs_impact.csv"
    impact.to_csv(impact_path, index=False)

    # Highlight VIX / sentiment rows
    highlight = impact[impact["feature"].isin(["vix_t", "sentiment_proxy"])].copy()
    highlight_path = REPORT_DIR / "shap_vix_sentiment_highlight.csv"
    highlight.to_csv(highlight_path, index=False)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_exp, feature_names=feature_names, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "shap_summary_week7.png", dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    top = impact.head(12)
    plt.barh(top["feature"][::-1], top["mean_abs_shap"][::-1])
    plt.xlabel("Mean |SHAP|")
    plt.title("Top features by mean absolute SHAP value (Week 7)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "shap_mean_abs_bar.png", dpi=160)
    plt.close()

    # Extreme scenarios: use last available row for S, sigma, r, VIX context; fall back to config
    last = frame.iloc[-1]
    s0 = float(last["close"])
    sigma0 = float(last["sigma_252d"])
    r_row = float(last["r"])

    scenario_df = extreme_scenario_table(s=s0, k=k, r=r_row, q=q, sigma=sigma0, t1=t1, t2=t2)
    scenario_cfg = extreme_scenario_table(
        s=float(m["s0"]),
        k=k,
        r=r_default,
        q=q,
        sigma=float(m["sigma"]),
        t1=t1,
        t2=t2,
    )
    scenario_df.to_csv(REPORT_DIR / "extreme_scenarios_last_row.csv", index=False)
    scenario_cfg.to_csv(REPORT_DIR / "extreme_scenarios_config_params.csv", index=False)

    def _mean_abs(name: str) -> Optional[float]:
        sub = impact.loc[impact["feature"] == name, "mean_abs_shap"]
        return float(sub.iloc[0]) if len(sub) else None

    summary = {
        "gbdt_test_metrics": gbdt_metrics,
        "shap_background_size": int(n_bg),
        "shap_explain_size": int(n_explain),
        "mean_abs_shap_vix_t": _mean_abs("vix_t"),
        "mean_abs_shap_sentiment": _mean_abs("sentiment_proxy"),
        "artifacts": {
            "shap_mean_abs_impact": str(impact_path.relative_to(PROJECT_ROOT)),
            "shap_vix_sentiment": str(highlight_path.relative_to(PROJECT_ROOT)),
            "extreme_last_row": "data/reports/week7/extreme_scenarios_last_row.csv",
            "extreme_config": "data/reports/week7/extreme_scenarios_config_params.csv",
        },
    }
    with open(REPORT_DIR / "week7_sensitivity_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Week 7 sensitivity analysis completed.")
    print(f"  SHAP table: {impact_path}")
    print(f"  Scenarios:  {REPORT_DIR / 'extreme_scenarios_last_row.csv'}")


if __name__ == "__main__":
    main()
