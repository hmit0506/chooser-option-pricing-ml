"""
Real-time / near-real-time market data refresh for the pricing pipeline.

Fetches the latest JPM OHLCV, VIX, and optionally Treasury (FRED) series,
merges with existing raw Parquet files, and writes updated snapshots.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_YAHOO = PROJECT_ROOT / "data" / "raw" / "yahoo_finance"


def _ensure_tz_naive_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if hasattr(out.index, "tz") and out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    out.index = pd.to_datetime(out.index).normalize()
    return out


def fetch_latest_yahoo_snapshot(
    ticker: str = "JPM",
    vix_symbol: str = "^VIX",
    lookback_days: int = 30,
) -> Dict[str, pd.DataFrame]:
    """
    Download recent daily bars for equity and VIX (no API key required).

    Args:
        ticker: Underlying symbol (default JPM).
        vix_symbol: Volatility index symbol.
        lookback_days: History window to request.

    Returns:
        Dict with keys 'equity' and 'vix', each a DataFrame indexed by date.
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days + 5)
    eq = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
    vx = yf.Ticker(vix_symbol).history(start=start, end=end, auto_adjust=True)
    if eq.empty or vx.empty:
        raise ValueError("Yahoo Finance returned empty history; check connectivity or symbols.")
    return {"equity": _ensure_tz_naive_index(eq), "vix": _ensure_tz_naive_index(vx)}


def merge_and_save_parquet(
    new_df: pd.DataFrame,
    parquet_path: Path,
    csv_path: Optional[Path] = None,
) -> int:
    """
    Merge new rows into an existing Parquet file (by index), or create new.

    Returns:
        Number of rows written.
    """
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    if parquet_path.exists():
        old = pd.read_parquet(parquet_path)
        if hasattr(old.index, "tz") and old.index.tz is not None:
            old.index = old.index.tz_localize(None)
        old.index = pd.to_datetime(old.index).normalize()
        combined = pd.concat([old, new_df])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = new_df.sort_index()
    combined.to_parquet(parquet_path)
    if csv_path is not None:
        combined.to_csv(csv_path)
    return len(combined)


def update_market_data_raw(
    ticker: str = "JPM",
    lookback_days: int = 60,
    raw_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Auto-update raw Yahoo Finance files used by preprocessing.

    Merges fresh history into:
      - {ticker}_daily_ohlcv.parquet / .csv
      - VIX_daily.parquet / .csv

    Environment:
        TARGET_TICKER: overrides default ticker if set.

    Returns:
        Summary dict with paths and row counts.
    """
    raw_dir = raw_dir or RAW_YAHOO
    ticker = os.getenv("TARGET_TICKER", ticker)
    snap = fetch_latest_yahoo_snapshot(ticker=ticker, lookback_days=lookback_days)

    eq_path = raw_dir / f"{ticker}_daily_ohlcv.parquet"
    eq_csv = raw_dir / f"{ticker}_daily_ohlcv.csv"
    vx_path = raw_dir / "VIX_daily.parquet"
    vx_csv = raw_dir / "VIX_daily.csv"

    n_eq = merge_and_save_parquet(snap["equity"], eq_path, eq_csv)
    n_vx = merge_and_save_parquet(snap["vix"], vx_path, vx_csv)

    last_eq = snap["equity"].index.max()
    last_vx = snap["vix"].index.max()

    return {
        "ticker": ticker,
        "equity_parquet": str(eq_path.relative_to(PROJECT_ROOT)),
        "vix_parquet": str(vx_path.relative_to(PROJECT_ROOT)),
        "equity_rows_total": n_eq,
        "vix_rows_total": n_vx,
        "last_equity_date": str(last_eq.date()) if last_eq is not None else None,
        "last_vix_date": str(last_vx.date()) if last_vx is not None else None,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def get_latest_quote_summary(ticker: str = "JPM") -> Dict[str, float]:
    """
    Lightweight snapshot: last close for equity and VIX (for UI / API).
    """
    ticker = os.getenv("TARGET_TICKER", ticker)
    snap = fetch_latest_yahoo_snapshot(ticker=ticker, lookback_days=10)
    eq = snap["equity"]
    vx = snap["vix"]
    return {
        "equity_close": float(eq["Close"].iloc[-1]),
        "vix_close": float(vx["Close"].iloc[-1]),
        "as_of_equity": str(eq.index[-1].date()),
        "as_of_vix": str(vx.index[-1].date()),
    }


__all__ = [
    "fetch_latest_yahoo_snapshot",
    "merge_and_save_parquet",
    "update_market_data_raw",
    "get_latest_quote_summary",
]
