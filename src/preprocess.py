"""
Week 2 Data Preprocessing Pipeline for Chooser Option Pricing Model.
Loads raw data, cleans, engineers features, and saves processed dataset.
Run: python src/preprocess.py   (from project root)
"""

import sys
from pathlib import Path

# Ensure project root is on path for imports
_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.data.loaders import load_raw_data
from src.features.feature_engineering import add_all_features, handle_outliers_iqr

# Output paths
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PARQUET = PROCESSED_DIR / "processed_dataset.parquet"
OUTPUT_CSV = PROCESSED_DIR / "processed_dataset.csv"


def clean_and_align(data: dict, ticker: str = "JPM") -> pd.DataFrame:
    """
    Merge raw datasets onto a unified daily index. Handle missing values and alignment.

    Uses JPM trading days as base calendar. Forward-fills VIX and Treasury on non-trading days.
    Treasury is optional (when FRED_API_KEY not set); uses default 0.04 if missing.
    """
    stock = data["stock"]
    vix = data["vix"]
    treasury = data.get("treasury")
    dividends = data.get("dividends")

    # Base index: union of stock and vix dates (trading days)
    base_dates = stock.index.union(vix.index).sort_values().unique()
    base_index = pd.DatetimeIndex(base_dates)

    # Build merged DataFrame
    df = pd.DataFrame(index=base_index)

    # JPM: use Close, High, Low, Volume, Dividends
    jpm_cols = ["Open", "High", "Low", "Close", "Volume", "Dividends"]
    for col in jpm_cols:
        if col in stock.columns:
            df[col] = stock[col].reindex(base_index).ffill()

    # Merge dividends: expand to daily (0 on non-dividend days)
    if dividends is not None and not dividends.empty:
        div_col = dividends.columns[0] if len(dividends.columns) > 0 else "Dividend"
        div_series = dividends[div_col].reindex(base_index, fill_value=0)
        # Overwrite Dividends from stock if we have explicit dividend dates
        df["Dividends"] = div_series

    # VIX close
    vix_close = vix["Close"].reindex(base_index).ffill()
    df["vix_close"] = vix_close

    # Treasury: DGS10 as risk-free rate proxy (convert % to decimal)
    # Use placeholder 0.04 when FRED data not available (e.g., CI without FRED_API_KEY)
    if treasury is not None:
        treasury_col = "DGS10" if "DGS10" in treasury.columns else "value"
        treasury_10y = treasury[treasury_col].reindex(base_index).ffill()
        df["treasury_10y"] = treasury_10y / 100.0  # 2.46 -> 0.0246
    else:
        df["treasury_10y"] = 0.04  # Default risk-free rate when FRED unavailable

    return df


def interpolate_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolate missing values. Use linear interpolation for numeric columns.
    Limit consecutive NaNs to avoid extrapolating too far.
    """
    out = df.copy()
    numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_cols:
        if out[col].isna().sum() > 0:
            out[col] = out[col].interpolate(method="linear", limit=5, limit_direction="both")
            # Remaining NaNs: forward then backward fill
            out[col] = out[col].ffill().bfill()

    return out


def run_pipeline(ticker: str = "JPM", verbose: bool = True) -> pd.DataFrame:
    """
    Run the full preprocessing pipeline: load -> clean -> features -> outlier handling -> save.

    Args:
        ticker: Stock ticker (e.g., JPM).
        verbose: If True, print shapes and sample info.

    Returns:
        Processed DataFrame.
    """
    if verbose:
        print("=" * 60)
        print("WEEK 2 PREPROCESSING PIPELINE")
        print("=" * 60)

    # 1. Load
    if verbose:
        print("\n[1/5] Loading raw data...")
    data = load_raw_data(ticker)
    if verbose:
        treasury_shape = data["treasury"].shape if data.get("treasury") is not None else "N/A (using default)"
        print(f"  Stock: {data['stock'].shape}, VIX: {data['vix'].shape}, Treasury: {treasury_shape}")

    # 2. Clean & align
    if verbose:
        print("\n[2/5] Merging and aligning to daily index...")
    df = clean_and_align(data, ticker)
    if verbose:
        print(f"  Merged shape: {df.shape}")
        print(f"  Date range: {df.index.min().date()} to {df.index.max().date()}")

    # 3. Interpolate
    if verbose:
        print("\n[3/5] Interpolating missing values...")
    df = interpolate_missing(df)
    if verbose:
        print(f"  NaNs after interpolate: {df.isna().sum().sum()}")

    # 4. Feature engineering
    if verbose:
        print("\n[4/5] Adding features...")
    df = add_all_features(df)
    feature_cols = [c for c in df.columns if c not in ["Open", "High", "Low", "Close", "Volume", "Dividends"]]
    if verbose:
        print(f"  Total columns: {len(df.columns)}, Feature columns: {len(feature_cols)}")
        print(f"  Features: {feature_cols}")

    # 5. Outlier handling (IQR)
    if verbose:
        print("\n[5/5] Handling outliers (IQR winsorization)...")
    df = handle_outliers_iqr(df)

    # Drop initial rows with NaNs from rolling windows (no look-ahead)
    min_valid = df["vol_252d"].first_valid_index()
    if min_valid is not None:
        df = df.loc[min_valid:]
    if verbose:
        print(f"  Final shape after trimming: {df.shape}")

    # 6. Save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\nSaving outputs...")

    df.to_csv(OUTPUT_CSV)
    if verbose:
        print(f"  CSV: {OUTPUT_CSV}")

    try:
        df.to_parquet(OUTPUT_PARQUET, compression="snappy", index=True)
        if verbose:
            print(f"  Parquet: {OUTPUT_PARQUET}")
    except Exception as e:
        if verbose:
            print(f"  [WARN] Parquet save failed: {e}. Use CSV output.")

    if verbose:
        print("\n" + "=" * 60)
        print("SAMPLE ROWS (last 3)")
        print("=" * 60)
        print(df.tail(3).to_string())
        print("\n[OK] Pipeline complete.")

    return df


def main() -> int:
    """Entry point for script execution."""
    try:
        run_pipeline(verbose=True)
        return 0
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Run scripts/data_collection/collect_all.py first to populate data/raw/")
        return 1
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
