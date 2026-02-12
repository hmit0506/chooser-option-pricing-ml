"""
Data loaders for raw Yahoo Finance and FRED datasets.
Loads from data/raw/yahoo_finance/ and data/raw/fred/.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

# Project root: parent of src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_YAHOO_DIR = PROJECT_ROOT / "data" / "raw" / "yahoo_finance"
RAW_FRED_DIR = PROJECT_ROOT / "data" / "raw" / "fred"


def _load_file(path: Path, prefer_parquet: bool = True) -> pd.DataFrame:
    """
    Load a single file, preferring parquet over csv.

    Args:
        path: Path without extension.
        prefer_parquet: If True, try parquet first.

    Returns:
        Loaded DataFrame.
    """
    parquet_path = path.with_suffix(".parquet")
    csv_path = path.with_suffix(".csv")

    if prefer_parquet and parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path)
        except Exception:
            pass

    if csv_path.exists():
        return pd.read_csv(csv_path, index_col=0, parse_dates=True)

    raise FileNotFoundError(f"Neither {parquet_path} nor {csv_path} found")


def load_yahoo_data(ticker: str = "JPM") -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Load Yahoo Finance data: stock OHLCV, VIX, dividends.

    Args:
        ticker: Stock ticker (e.g., JPM).

    Returns:
        Tuple of (stock_df, vix_df, dividends_df). dividends_df may be None if empty.
    """
    stock_df = _load_file(RAW_YAHOO_DIR / f"{ticker}_daily_ohlcv")
    vix_df = _load_file(RAW_YAHOO_DIR / "VIX_daily")

    dividends_path = RAW_YAHOO_DIR / f"{ticker}_dividends"
    dividends_df = None
    if dividends_path.with_suffix(".parquet").exists() or dividends_path.with_suffix(".csv").exists():
        div = _load_file(dividends_path)
        if not div.empty:
            dividends_df = div

    return stock_df, vix_df, dividends_df


def load_fred_data() -> pd.DataFrame:
    """Load FRED treasury rates (combined DGS3MO, DGS10, FEDFUNDS)."""
    combined_path = RAW_FRED_DIR / "treasury_rates_combined"
    return _load_file(combined_path)


def load_raw_data(ticker: str = "JPM") -> dict:
    """
    Load all raw data into a single dict for pipeline use.

    Args:
        ticker: Stock ticker (e.g., JPM).

    Returns:
        Dict with keys: stock, vix, dividends (optional), treasury.
    """
    stock_df, vix_df, dividends_df = load_yahoo_data(ticker)
    treasury_df = load_fred_data()

    # Normalize date index: strip timezone, use date only
    for df in [stock_df, vix_df]:
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index = pd.to_datetime(df.index).normalize()

    if dividends_df is not None:
        if hasattr(dividends_df.index, "tz") and dividends_df.index.tz is not None:
            dividends_df.index = dividends_df.index.tz_localize(None)
        dividends_df.index = pd.to_datetime(dividends_df.index).normalize()

    treasury_df.index = pd.to_datetime(treasury_df.index).normalize()

    return {
        "stock": stock_df,
        "vix": vix_df,
        "dividends": dividends_df,
        "treasury": treasury_df,
    }
