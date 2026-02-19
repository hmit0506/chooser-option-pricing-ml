# Week 2 Weekly Report – Data Preprocessing & Feature Engineering

**Project:** Avalok Capital Quantitative Research & Trading - Advanced Chooser Option Pricing Model with Real-World Data & Machine Learning  
**Author:** Shi Qian  
**Date:** February 11, 2026  
**Week:** 2  
**Version:** 1.0  

---

## Summary

Week 2 focused on building the data preprocessing pipeline and feature engineering layer to support the BSM chooser model and future ML work. All planned deliverables for this milestone were completed.

---

## Completed Work

### 1. Automated Preprocessing Pipeline

- **Location:** `src/preprocess.py` (runnable as `python src/preprocess.py` from project root)
- **Flow:** Load raw data → clean & align → interpolate missing values → add features → IQR outlier handling → save
- **Outputs:** `data/processed/processed_dataset.parquet` and `processed_dataset.csv`
- **Design choices:**
  - Uses union of JPM and VIX trading days as base calendar
  - FRED Treasury data optional (defaults to 0.04 when `FRED_API_KEY` not set, e.g. in CI)
  - Time-series aware: forward-fill, linear interpolation (limit=5), no look-ahead in rolling windows

### 2. Data Loading Module

- **Location:** `src/data/loaders.py`
- Loads from `data/raw/yahoo_finance/` (JPM OHLCV, VIX, dividends) and `data/raw/fred/` (treasury rates)
- Prefers Parquet, falls back to CSV
- FRED loader returns `None` if files are missing; pipeline continues with Yahoo data only

### 3. Feature Engineering (≥10 Features)

- **Location:** `src/features/feature_engineering.py`
- **Traditional features:** `log_return`, `vol_21d`, `vol_63d`, `vol_252d`, `vix_close`, `treasury_10y`, `dividend_yield_proxy`, `volume_ma_21d`, `high_low_range_pct`
- **Advanced features:** `vix_jpm_corr_63d`, `treasury_momentum_21d`, `sentiment_proxy`
- **Rationale:** Supports BSM inputs (S, σ, r, q) and ML volatility prediction

### 4. Feature Documentation

- **Location:** `docs/feature_engineering.md`
- Table of 12 features: name, type, formula, rationale
- Data sources and implementation notes

### 5. CI/CD (GitHub Actions)

- **Location:** `.github/workflows/preprocessing.yml`
- **Steps:** Collect raw data → run preprocessing → upload processed artifact
- **Triggers:** Daily schedule, manual dispatch, push to `main` on relevant paths
- **Notes:** Yahoo Finance collected without API key; FRED requires `FRED_API_KEY` secret for full data

---

## Deliverables Checklist

| Deliverable | Status |
|-------------|--------|
| Cleaned structured dataset (parquet + CSV) | Done |
| ≥10 feature columns | Done (12 features) |
| Modular preprocessing pipeline | Done |
| Feature engineering documentation | Done |
| Basic GitHub Actions workflow | Done |

---

## Next Steps (Week 3+)

### Model Implementation

1. **BSM chooser model:** Implement the BSM-based chooser option pricing formula from the reference paper in Python.
2. **Parameter configuration:** Set strike = $150 and T2 = 1 year to match paper parameters.
3. **Initial validation:** Compare simulated outputs with the paper’s reported results to confirm correctness of the implementation.

These steps will establish the baseline model used for later ML comparison and real-world data validation.

---

## References

- Week 1: `docs/weekly_reports/data_requirement_specification.md`
- Feature details: `docs/feature_engineering.md`
- Project Outline: Avalok Capital PDF
