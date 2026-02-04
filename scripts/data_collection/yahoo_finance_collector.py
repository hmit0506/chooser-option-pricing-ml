"""
Yahoo Finance Data Collector
Collects stock prices, VIX index, and dividend data from Yahoo Finance.
"""

import yfinance as yf
import pandas as pd
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import warnings

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

# Configuration from environment
DATA_START_DATE = os.getenv('DATA_START_DATE', '2018-01-01')
DATA_END_DATE = os.getenv('DATA_END_DATE', '2024-12-31')
TARGET_TICKER = os.getenv('TARGET_TICKER', 'JPM')
OUTPUT_DIR = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'yahoo_finance'


class YahooFinanceCollector:
    """Collector for Yahoo Finance data."""
    
    def __init__(self, ticker: str, start_date: str, end_date: str, output_dir: Path):
        """
        Initialize collector.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'JPM')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            output_dir: Directory to save collected data
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def collect_stock_data(self) -> pd.DataFrame:
        """Collect daily OHLCV data for the target ticker."""
        print(f"Collecting {self.ticker} stock data from {self.start_date} to {self.end_date}...")
        
        ticker_obj = yf.Ticker(self.ticker)
        data = ticker_obj.history(start=self.start_date, end=self.end_date, auto_adjust=True)
        
        if data.empty:
            raise ValueError(f"No data retrieved for {self.ticker}")
        
        print(f"  Retrieved {len(data)} days of data")
        print(f"  Date range: {data.index.min().date()} to {data.index.max().date()}")
        print(f"  Columns: {', '.join(data.columns)}")
        
        return data
    
    def collect_vix_data(self) -> pd.DataFrame:
        """Collect VIX index data."""
        print(f"\nCollecting VIX index data from {self.start_date} to {self.end_date}...")
        
        vix_ticker = yf.Ticker("^VIX")
        data = vix_ticker.history(start=self.start_date, end=self.end_date, auto_adjust=True)
        
        if data.empty:
            raise ValueError("No VIX data retrieved")
        
        print(f"  Retrieved {len(data)} days of VIX data")
        print(f"  Date range: {data.index.min().date()} to {data.index.max().date()}")
        
        return data
    
    def collect_dividend_data(self) -> pd.DataFrame:
        """Collect dividend data for the target ticker."""
        print(f"\nCollecting dividend data for {self.ticker}...")
        
        ticker_obj = yf.Ticker(self.ticker)
        dividends = ticker_obj.dividends
        
        if dividends.empty:
            print("  No dividend data found")
            return pd.DataFrame()
        
        # Filter by date range
        dividends = dividends[(dividends.index >= self.start_date) & (dividends.index <= self.end_date)]
        
        if dividends.empty:
            print(f"  No dividends in date range {self.start_date} to {self.end_date}")
            return pd.DataFrame()
        
        print(f"  Retrieved {len(dividends)} dividend payments")
        print(f"  Date range: {dividends.index.min().date()} to {dividends.index.max().date()}")
        
        return dividends.to_frame(name='Dividend')
    
    def save_data(self, data: pd.DataFrame, filename: str):
        """Save data to CSV and Parquet formats."""
        csv_path = self.output_dir / f"{filename}.csv"
        parquet_path = self.output_dir / f"{filename}.parquet"
        
        # Save as CSV
        data.to_csv(csv_path)
        print(f"  Saved CSV: {csv_path}")
        
        # Save as Parquet (more efficient for large datasets)
        data.to_parquet(parquet_path, compression='snappy')
        print(f"  Saved Parquet: {parquet_path}")
    
    def run(self):
        """Run all data collection tasks."""
        print("="*60)
        print("YAHOO FINANCE DATA COLLECTION")
        print("="*60)
        print(f"Ticker: {self.ticker}")
        print(f"Date Range: {self.start_date} to {self.end_date}")
        print(f"Output Directory: {self.output_dir}")
        print("="*60)
        
        try:
            # Collect stock data
            stock_data = self.collect_stock_data()
            self.save_data(stock_data, f"{self.ticker}_daily_ohlcv")
            
            # Collect VIX data
            vix_data = self.collect_vix_data()
            self.save_data(vix_data, "VIX_daily")
            
            # Collect dividend data
            dividend_data = self.collect_dividend_data()
            if not dividend_data.empty:
                self.save_data(dividend_data, f"{self.ticker}_dividends")
            
            print("\n" + "="*60)
            print("[OK] Yahoo Finance data collection completed successfully!")
            print("="*60)
            
            return True
            
        except Exception as e:
            print(f"\n[FAIL] Data collection failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main function."""
    collector = YahooFinanceCollector(
        ticker=TARGET_TICKER,
        start_date=DATA_START_DATE,
        end_date=DATA_END_DATE,
        output_dir=OUTPUT_DIR
    )
    
    success = collector.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
