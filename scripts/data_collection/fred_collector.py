"""
FRED (Federal Reserve Economic Data) Collector
Collects Treasury rates and macroeconomic indicators from FRED API.
"""

import pandas as pd
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import warnings
import time

# Load environment variables
try:
    from dotenv import load_dotenv
    project_root = Path(__file__).parent.parent.parent
    env_file = project_root / '.env'
    if env_file.exists():
        load_dotenv(env_file, override=False)
except ImportError:
    pass

warnings.filterwarnings('ignore')

# Try to import FRED API
try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    print("ERROR: fredapi not installed. Install with: pip install fredapi")
    FRED_AVAILABLE = False
    sys.exit(1)

# Configuration from environment
DATA_START_DATE = os.getenv('DATA_START_DATE', '2018-01-01')
DATA_END_DATE = os.getenv('DATA_END_DATE', '2024-12-31')
FRED_API_KEY = os.getenv('FRED_API_KEY')
RISK_FREE_RATE_SERIES = os.getenv('RISK_FREE_RATE_SERIES', 'DGS3MO')
TREASURY_10Y_SERIES = os.getenv('TREASURY_10Y_SERIES', 'DGS10')
FED_FUNDS_SERIES = os.getenv('FED_FUNDS_SERIES', 'FEDFUNDS')
OUTPUT_DIR = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'fred'


class FREDCollector:
    """Collector for FRED economic data."""
    
    def __init__(self, api_key: str, start_date: str, end_date: str, output_dir: Path):
        """
        Initialize collector.
        
        Args:
            api_key: FRED API key
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            output_dir: Directory to save collected data
        """
        if not api_key:
            raise ValueError("FRED_API_KEY is required. Set it in .env file or environment variable.")
        
        self.client = Fred(api_key=api_key)
        self.start_date = start_date
        self.end_date = end_date
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def collect_series(self, series_id: str, series_name: str) -> Optional[pd.Series]:
        """
        Collect a single FRED series.
        
        Args:
            series_id: FRED series ID (e.g., 'DGS3MO')
            series_name: Human-readable name for the series
        
        Returns:
            pandas Series with the data, or None if failed
        """
        print(f"\nCollecting {series_name} ({series_id})...")
        
        try:
            data = self.client.get_series(series_id, observation_start=self.start_date, observation_end=self.end_date)
            
            if data.empty:
                print(f"  [WARN] No data retrieved for {series_id}")
                return None
            
            print(f"  Retrieved {len(data)} data points")
            print(f"  Date range: {data.index.min().date()} to {data.index.max().date()}")
            print(f"  Value range: {data.min():.4f} to {data.max():.4f}")
            
            # Get series info
            try:
                info = self.client.get_series_info(series_id)
                print(f"  Title: {info.get('title', 'N/A')}")
                print(f"  Frequency: {info.get('frequency', 'N/A')}")
            except Exception as e:
                print(f"  [WARN] Could not fetch series info: {e}")
            
            # Rate limiting (120 calls/minute = ~0.5 seconds between calls)
            time.sleep(0.6)
            
            return data
            
        except Exception as e:
            print(f"  [FAIL] Failed to collect {series_id}: {e}")
            return None
    
    def save_series(self, data: pd.Series, filename: str):
        """Save series data to CSV and Parquet formats."""
        csv_path = self.output_dir / f"{filename}.csv"
        parquet_path = self.output_dir / f"{filename}.parquet"
        
        # Convert Series to DataFrame for easier handling
        df = data.to_frame(name='value')
        df.index.name = 'date'
        
        # Save as CSV
        df.to_csv(csv_path)
        print(f"  Saved CSV: {csv_path}")
        
        # Save as Parquet
        df.to_parquet(parquet_path, compression='snappy')
        print(f"  Saved Parquet: {parquet_path}")
    
    def collect_all_series(self, series_config: List[tuple]) -> dict:
        """
        Collect multiple FRED series.
        
        Args:
            series_config: List of (series_id, series_name) tuples
        
        Returns:
            Dictionary mapping series_id to collected data
        """
        results = {}
        
        for series_id, series_name in series_config:
            data = self.collect_series(series_id, series_name)
            if data is not None:
                results[series_id] = data
                # Save individual series
                self.save_series(data, series_id)
        
        return results
    
    def run(self):
        """Run all data collection tasks."""
        print("="*60)
        print("FRED DATA COLLECTION")
        print("="*60)
        print(f"Date Range: {self.start_date} to {self.end_date}")
        print(f"Output Directory: {self.output_dir}")
        print("="*60)
        
        # Define series to collect
        series_to_collect = [
            (RISK_FREE_RATE_SERIES, "3-Month Treasury Rate"),
            (TREASURY_10Y_SERIES, "10-Year Treasury Rate"),
            (FED_FUNDS_SERIES, "Federal Funds Rate"),
        ]
        
        try:
            results = self.collect_all_series(series_to_collect)
            
            # Create combined dataset if multiple series collected
            if len(results) > 1:
                print("\nCreating combined dataset...")
                combined_df = pd.DataFrame(results)
                combined_df.index.name = 'date'
                
                combined_csv = self.output_dir / "treasury_rates_combined.csv"
                combined_parquet = self.output_dir / "treasury_rates_combined.parquet"
                
                combined_df.to_csv(combined_csv)
                combined_df.to_parquet(combined_parquet, compression='snappy')
                
                print(f"  Saved combined CSV: {combined_csv}")
                print(f"  Saved combined Parquet: {combined_parquet}")
            
            print("\n" + "="*60)
            print(f"[OK] FRED data collection completed! Collected {len(results)} series")
            print("="*60)
            
            return True
            
        except Exception as e:
            print(f"\n[FAIL] Data collection failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main function."""
    if not FRED_API_KEY:
        print("ERROR: FRED_API_KEY not found in environment variables or .env file")
        print("Please set FRED_API_KEY in your .env file")
        sys.exit(1)
    
    collector = FREDCollector(
        api_key=FRED_API_KEY,
        start_date=DATA_START_DATE,
        end_date=DATA_END_DATE,
        output_dir=OUTPUT_DIR
    )
    
    success = collector.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
