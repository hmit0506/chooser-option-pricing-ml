"""
Master script to run all data collectors.
Collects data from Yahoo Finance and FRED APIs.
"""

import sys
from pathlib import Path

# Add scripts directory to path
scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir))

from yahoo_finance_collector import YahooFinanceCollector
from fred_collector import FREDCollector
import os

# Load environment variables
try:
    from dotenv import load_dotenv
    project_root = Path(__file__).parent.parent.parent
    env_file = project_root / '.env'
    if env_file.exists():
        load_dotenv(env_file, override=False)
except ImportError:
    pass

# Configuration
DATA_START_DATE = os.getenv('DATA_START_DATE', '2018-01-01')
DATA_END_DATE = os.getenv('DATA_END_DATE', '2024-12-31')
TARGET_TICKER = os.getenv('TARGET_TICKER', 'JPM')
FRED_API_KEY = os.getenv('FRED_API_KEY')

YAHOO_OUTPUT_DIR = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'yahoo_finance'
FRED_OUTPUT_DIR = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'fred'


def main():
    """Run all data collectors."""
    print("="*70)
    print("INITIAL DATA COLLECTION - 2018-2024")
    print("="*70)
    print(f"Date Range: {DATA_START_DATE} to {DATA_END_DATE}")
    print(f"Target Ticker: {TARGET_TICKER}")
    print("="*70)
    
    success_count = 0
    total_count = 0
    
    # Collect Yahoo Finance data
    print("\n[1/2] Collecting Yahoo Finance data...")
    print("-" * 70)
    try:
        yahoo_collector = YahooFinanceCollector(
            ticker=TARGET_TICKER,
            start_date=DATA_START_DATE,
            end_date=DATA_END_DATE,
            output_dir=YAHOO_OUTPUT_DIR
        )
        if yahoo_collector.run():
            success_count += 1
        total_count += 1
    except Exception as e:
        print(f"[FAIL] Yahoo Finance collection failed: {e}")
        total_count += 1
    
    # Collect FRED data
    print("\n[2/2] Collecting FRED data...")
    print("-" * 70)
    if FRED_API_KEY:
        try:
            fred_collector = FREDCollector(
                api_key=FRED_API_KEY,
                start_date=DATA_START_DATE,
                end_date=DATA_END_DATE,
                output_dir=FRED_OUTPUT_DIR
            )
            if fred_collector.run():
                success_count += 1
            total_count += 1
        except Exception as e:
            print(f"[FAIL] FRED collection failed: {e}")
            total_count += 1
    else:
        print("[SKIP] FRED_API_KEY not found, skipping FRED data collection")
    
    # Summary
    print("\n" + "="*70)
    print("COLLECTION SUMMARY")
    print("="*70)
    print(f"Completed: {success_count}/{total_count} collectors")
    
    if success_count == total_count:
        print("[OK] All data collection completed successfully!")
        print(f"\nData saved to:")
        print(f"  - Yahoo Finance: {YAHOO_OUTPUT_DIR}")
        print(f"  - FRED: {FRED_OUTPUT_DIR}")
        return 0
    else:
        print(f"[WARN] {total_count - success_count} collector(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
