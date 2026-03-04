"""
Dataset builders and time-series splits for ML models.
"""

from typing import Tuple

import numpy as np
import pandas as pd

from pathlib import Path

from src.models import rubinstein_chooser, realized_proxy_pv


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_base_frame(ticker: str = "JPM") -> pd.DataFrame:
    """
    Build a base daily DataFrame with aligned JPM, VIX and DGS10 series
    and core engineered features used by Week 4/5 models.
    """
    yahoo_dir = PROJECT_ROOT / "data" / "raw" / "yahoo_finance"
    fred_dir = PROJECT_ROOT / "data" / "raw" / "fred"

    jpm = pd.read_parquet(yahoo_dir / f"{ticker}_daily_ohlcv.parquet")
    vix = pd.read_parquet(yahoo_dir / "VIX_daily.parquet")

    dgs10_path = fred_dir / "DGS10.parquet"
    dgs10 = pd.read_parquet(dgs10_path) if dgs10_path.exists() else None

    for df in [jpm, vix]:
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index = pd.to_datetime(df.index).normalize()

    if dgs10 is not None:
        dgs10.index = pd.to_datetime(dgs10.index).normalize()

    frame = pd.DataFrame(index=jpm.index.copy())
    frame["close"] = jpm["Close"].astype(float)
    frame["vix"] = vix["Close"].reindex(frame.index).ffill().bfill().astype(float)

    # risk-free rate from DGS10 (percent -> decimal) or fallback to config-level default later
    if dgs10 is not None:
        dgs = dgs10.iloc[:, 0].reindex(frame.index).ffill().bfill().astype(float)
        frame["r"] = np.where(dgs > 1.0, dgs / 100.0, dgs)

    frame["log_ret"] = np.log(frame["close"] / frame["close"].shift(1))
    frame["sigma_252d"] = frame["log_ret"].rolling(252, min_periods=200).std() * np.sqrt(252)

    # VIX-based sentiment proxy
    vix_min = frame["vix"].rolling(252, min_periods=63).min()
    vix_max = frame["vix"].rolling(252, min_periods=63).max()
    spread = (vix_max - vix_min).replace(0, np.nan)
    frame["sentiment_proxy"] = 1 - (frame["vix"] - vix_min) / spread

    frame = frame.dropna(subset=["close", "vix", "sigma_252d"])
    return frame


def build_volatility_dataset(
    frame: pd.DataFrame,
    horizon_days: int,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Build (X, y, dates) for volatility forecasting.

    y is next-horizon realized volatility, annualized, computed from future log returns.
    """
    df = frame.copy()
    # future realized vol label
    log_ret = df["log_ret"].values
    realized = np.full_like(log_ret, np.nan, dtype=float)

    for i in range(len(df) - horizon_days):
        window = log_ret[i + 1 : i + 1 + horizon_days]
        if np.any(~np.isfinite(window)):
            continue
        realized[i] = window.std() * np.sqrt(252.0)

    df["realized_vol_label"] = realized
    df = df.dropna(subset=["realized_vol_label", "sigma_252d"])

    feature_cols = ["close", "vix", "r", "sigma_252d", "sentiment_proxy"]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].values
    y = df["realized_vol_label"].values
    dates = df.index.to_numpy()
    return pd.DataFrame(X, index=dates, columns=feature_cols), y, dates


def build_pricing_dataset(
    frame: pd.DataFrame,
    k: float,
    r_default: float,
    q: float,
    t1_years: float,
    t2_years: float,
    t1_days: int,
    t2_days: int,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (X, y, dates, bsm_price) for pricing / residual modeling.

    y is proxy-actual PV (discounted realized payoff). BSM prices are computed
    via Rubinstein closed form using rolling sigma_252d and r_t (or fallback).
    """
    df = frame.copy()
    if "r" not in df.columns:
        df["r"] = r_default

    dates = df.index.to_list()
    rows = []

    for i in range(len(dates) - t2_days):
        t = dates[i]
        t1_date = dates[i + t1_days]
        t2_date = dates[i + t2_days]

        s_t = float(df.iloc[i]["close"])
        s_t1 = float(df.iloc[i + t1_days]["close"])
        s_t2 = float(df.iloc[i + t2_days]["close"])
        sigma_t = float(df.iloc[i]["sigma_252d"])
        r_t = float(df.iloc[i]["r"])
        vix_t = float(df.iloc[i]["vix"])
        senti_t = float(df.iloc[i]["sentiment_proxy"]) if pd.notna(df.iloc[i]["sentiment_proxy"]) else np.nan

        if not np.isfinite(sigma_t) or sigma_t <= 0:
            continue

        bsm_price = rubinstein_chooser(s_t, k, r_t, q, sigma_t, t1_years, t2_years)
        actual_proxy = realized_proxy_pv(
            s_t1_realized=s_t1,
            s_t2_realized=s_t2,
            k=k,
            r=r_t,
            t2=t2_years,
            use_proper_rule=False,
        )

        rows.append(
            {
                "date": t,
                "t1_date": t1_date,
                "t2_date": t2_date,
                "s_t": s_t,
                "s_t1": s_t1,
                "s_t2": s_t2,
                "sigma_t": sigma_t,
                "r_t": r_t,
                "vix_t": vix_t,
                "sentiment_proxy": senti_t,
                "bsm_price": bsm_price,
                "actual_proxy_pv": actual_proxy,
            }
        )

    bt = pd.DataFrame(rows).set_index("date")

    feature_cols = ["s_t", "vix_t", "r_t", "sigma_t", "sentiment_proxy"]
    X = bt[feature_cols].values
    y = bt["actual_proxy_pv"].values
    bsm_prices = bt["bsm_price"].values
    dates_out = bt.index.to_numpy()

    return pd.DataFrame(X, index=dates_out, columns=feature_cols), y, dates_out, bsm_prices


def time_series_split(
    X: pd.DataFrame,
    y: np.ndarray,
    dates: np.ndarray,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
):
    """
    Simple time-series split into train/val/test using chronological order.
    """
    n = len(dates)
    assert n == len(X) == len(y)
    idx = np.argsort(dates)
    X_sorted = X.iloc[idx].values if isinstance(X, pd.DataFrame) else X[idx]
    y_sorted = y[idx]
    dates_sorted = dates[idx]

    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    n_test = n - n_train - n_val

    X_train = X_sorted[:n_train]
    y_train = y_sorted[:n_train]
    X_val = X_sorted[n_train : n_train + n_val]
    y_val = y_sorted[n_train : n_train + n_val]
    X_test = X_sorted[n_train + n_val :]
    y_test = y_sorted[n_train + n_val :]
    dates_train = dates_sorted[:n_train]
    dates_val = dates_sorted[n_train : n_train + n_val]
    dates_test = dates_sorted[n_train + n_val :]

    return (
        (X_train, y_train, dates_train),
        (X_val, y_val, dates_val),
        (X_test, y_test, dates_test),
    )


__all__ = [
    "load_base_frame",
    "build_volatility_dataset",
    "build_pricing_dataset",
    "time_series_split",
]

