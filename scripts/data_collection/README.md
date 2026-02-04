# Data Collection Scripts

This directory contains scripts for collecting initial raw datasets (2018-2024) for the Chooser Option Pricing project.

## Scripts Overview

### 1. `api_tester.py`
Tests connectivity to Yahoo Finance, Alpha Vantage, and FRED APIs.
**Run this first** to verify API connections before data collection.

```bash
python scripts/data_collection/api_tester.py
```

### 2. `yahoo_finance_collector.py`
Collects data from Yahoo Finance:
- Stock prices (OHLCV) for target ticker (default: JPM)
- VIX index data
- Dividend data

**Usage:**
```bash
python scripts/data_collection/yahoo_finance_collector.py
```

**Output:** `data/raw/yahoo_finance/`
- `{TICKER}_daily_ohlcv.csv` / `.parquet`
- `VIX_daily.csv` / `.parquet`
- `{TICKER}_dividends.csv` / `.parquet`

### 3. `fred_collector.py`
Collects economic data from FRED API:
- 3-Month Treasury Rate (DGS3MO)
- 10-Year Treasury Rate (DGS10)
- Federal Funds Rate (FEDFUNDS)

**Requirements:** FRED_API_KEY must be set in `.env` file

**Usage:**
```bash
python scripts/data_collection/fred_collector.py
```

**Output:** `data/raw/fred/`
- `DGS3MO.csv` / `.parquet`
- `DGS10.csv` / `.parquet`
- `FEDFUNDS.csv` / `.parquet`
- `treasury_rates_combined.csv` / `.parquet`

### 4. `collect_all.py`
Master script that runs all collectors sequentially.

**Usage:**
```bash
python scripts/data_collection/collect_all.py
```

## Configuration

All scripts read configuration from `.env` file in the project root:

```env
# Date Range
DATA_START_DATE=2018-01-01
DATA_END_DATE=2024-12-31

# Target Instrument
TARGET_TICKER=JPM

# FRED Series IDs
RISK_FREE_RATE_SERIES=DGS3MO
TREASURY_10Y_SERIES=DGS10
FED_FUNDS_SERIES=FEDFUNDS

# API Keys
FRED_API_KEY=your_fred_api_key_here
```

## Data Storage Structure

```
data/raw/
├── yahoo_finance/
│   ├── JPM_daily_ohlcv.csv
│   ├── JPM_daily_ohlcv.parquet
│   ├── VIX_daily.csv
│   ├── VIX_daily.parquet
│   ├── JPM_dividends.csv
│   └── JPM_dividends.parquet
└── fred/
    ├── DGS3MO.csv
    ├── DGS3MO.parquet
    ├── DGS10.csv
    ├── DGS10.parquet
    ├── FEDFUNDS.csv
    ├── FEDFUNDS.parquet
    ├── treasury_rates_combined.csv
    └── treasury_rates_combined.parquet
```

## Quick Start

1. **Set up environment:**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

2. **Test API connections:**
   ```bash
   python scripts/data_collection/api_tester.py
   ```

3. **Collect all data:**
   ```bash
   python scripts/data_collection/collect_all.py
   ```

## Notes

- All data files are saved in both CSV (human-readable) and Parquet (efficient) formats
- Data collection respects API rate limits automatically
- Yahoo Finance has no rate limits (free and unlimited)
- FRED API: 120 calls/minute (scripts include automatic delays)
- Collected data files are gitignored (see `.gitignore`)
