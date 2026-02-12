"""
Feature engineering for chooser option pricing model.
Adds traditional and advanced features with no look-ahead bias.
"""

import numpy as np
import pandas as pd
from typing import Optional

# Trading days per year for annualization
TRADING_DAYS = 252


def _align_daily_index(df: pd.DataFrame, date_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Reindex df to date_index, forward-filling where appropriate."""
    return df.reindex(date_index).ffill().bfill()


def add_traditional_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add traditional BSM-related features: returns, volatilities, VIX, Treasury, dividend proxy.

    No look-ahead: rolling windows use historical data only.
    """
    out = df.copy()

    # 1. Log returns (use Close for JPM)
    if "Close" in out.columns:
        out["log_return"] = np.log(out["Close"] / out["Close"].shift(1))

    # 2. Rolling volatilities (annualized, 21/63/252 days)
    if "log_return" in out.columns:
        for w in [21, 63, 252]:
            out[f"vol_{w}d"] = (
                out["log_return"].rolling(window=w, min_periods=w // 2).std() * np.sqrt(TRADING_DAYS)
            )

    # 3. VIX level (already in df as vix_close or similar after merge)
    # Handled in merge step - ensure vix_close exists

    # 4. Treasury 10Y (already in df as treasury_10y)
    # Handled in merge step

    # 5. Dividend yield proxy: rolling 252d sum of dividends / close (annualized fraction)
    if "Dividends" in out.columns and "Close" in out.columns:
        div_sum = out["Dividends"].fillna(0).rolling(252, min_periods=1).sum()
        out["dividend_yield_proxy"] = div_sum / out["Close"].replace(0, np.nan)
    elif "Close" in out.columns:
        out["dividend_yield_proxy"] = 0.0

    # 6. Volume 21d moving average
    if "Volume" in out.columns:
        out["volume_ma_21d"] = out["Volume"].rolling(21, min_periods=1).mean()

    # 7. High-Low range as % of close (intraday volatility proxy)
    if "High" in out.columns and "Low" in out.columns and "Close" in out.columns:
        out["high_low_range_pct"] = (out["High"] - out["Low"]) / out["Close"].replace(0, np.nan) * 100

    return out


def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add advanced features: VIX-JPM correlation, Treasury momentum, sentiment proxy.
    """
    out = df.copy()

    # 1. VIX-JPM rolling 63d correlation (returns vs VIX change)
    if "log_return" in out.columns and "vix_close" in out.columns:
        vix_change = out["vix_close"].pct_change()
        out["vix_jpm_corr_63d"] = out["log_return"].rolling(63, min_periods=30).corr(vix_change)
    else:
        out["vix_jpm_corr_63d"] = np.nan

    # 2. Treasury 10Y momentum (21d change, or ROC)
    if "treasury_10y" in out.columns:
        out["treasury_momentum_21d"] = out["treasury_10y"].diff(21)
    else:
        out["treasury_momentum_21d"] = np.nan

    # 3. Sentiment proxy: 1 - minmax_norm(VIX) -> [0, 1], high VIX = low sentiment
    if "vix_close" in out.columns:
        vix = out["vix_close"]
        vix_min = vix.rolling(252, min_periods=21).min()
        vix_max = vix.rolling(252, min_periods=21).max()
        # Avoid div by zero
        spread = (vix_max - vix_min).replace(0, np.nan)
        normed = (vix - vix_min) / spread
        out["sentiment_proxy"] = 1 - normed
    else:
        out["sentiment_proxy"] = np.nan

    return out


def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all traditional and advanced features.

    Expects df to have columns from merged data:
    - Close, High, Low, Volume, Dividends (from JPM)
    - vix_close (from VIX)
    - treasury_10y (from FRED DGS10)
    """
    df = add_traditional_features(df)
    df = add_advanced_features(df)
    return df


def handle_outliers_iqr(
    df: pd.DataFrame,
    columns: Optional[list] = None,
    factor: float = 1.5,
) -> pd.DataFrame:
    """
    Winsorize outliers using IQR method for specified columns.

    Args:
        df: Input DataFrame.
        columns: Columns to process. If None, use numeric feature columns only.
        factor: IQR multiplier (default 1.5).

    Returns:
        DataFrame with outliers winsorized.
    """
    feature_cols = [
        "log_return", "vol_21d", "vol_63d", "vol_252d",
        "vix_jpm_corr_63d", "treasury_momentum_21d", "sentiment_proxy",
        "dividend_yield_proxy", "volume_ma_21d", "high_low_range_pct",
    ]
    cols = columns or [c for c in feature_cols if c in df.columns]
    out = df.copy()

    for col in cols:
        if col not in out.columns:
            continue
        q1 = out[col].quantile(0.25)
        q3 = out[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        out[col] = out[col].clip(lower=lower, upper=upper)

    return out
