# Week 4 Weekly Report – Baseline Model Performance Evaluation

**Project:** Avalok Capital Quantitative Research & Trading - Advanced Chooser Option Pricing Model with Real-World Data & Machine Learning  
**Author:** Shi Qian  
**Date:** February 2026  
**Week:** 4  
**Version:** 1.0  

---

## Summary

Week 4 focused on validating the BSM chooser baseline with quantitative error metrics, documenting model limitations, and formalizing a benchmark for upcoming ML models.  
Since direct public CME chooser transaction prices were not available in the current setup, a reproducible historical realized proxy framework was used for evaluation.

---

## Completed Work

### 1. Error Metric Calculation (MAE / RMSE)

- **Main implementation:** `notebooks/week4_validation.ipynb`
- **Core method:**
  1. Predict chooser price at date `t` using Rubinstein closed-form under BSM.
  2. Use realized JPM path (`t+T1`, `t+T2`) to compute ex-post chooser payoff.
  3. Discount realized payoff back to `t` as `actual_proxy_pv`.
  4. Compute errors between `pred_price` and `actual_proxy_pv`.
- **Key outputs generated:**
  - `docs/week4_metrics_summary.csv`
  - `docs/week4_regime_metrics.csv`
  - `notebooks/week4_validation_plots.png`

### 2. Limitation Analysis

- **High-volatility regime analysis:** VIX-based split (`high_vol` vs `normal_vol`) with regime-wise MAE/RMSE.
- **Sentiment impact gap analysis:** proxy sentiment split (`low_sentiment` vs `normal_sentiment`) and error comparison.
- **Documented constraints:** direct CME chooser transaction data is not readily available in public/free pipeline; evaluation relies on a transparent proxy baseline.

### 3. Benchmark Establishment

- **Benchmark document:** `docs/bsm_benchmark.md`
- **Validation report (deliverable):** `docs/week4_validation_report.md`
- **Reusable model utilities updated:** `src/models/bsm_chooser.py`
  - `compute_error_metrics`
  - `realized_proxy_pv`
  - `vix_regime_label`
  - `summarize_metrics_by_regime`
- **Exports updated:** `src/models/__init__.py`

---

## Deliverables Checklist

| Deliverable | Status |
|---|---|
| Model validation report with error metrics | Done (`docs/week4_validation_report.md`) |
| Performance benchmark documentation | Done (`docs/bsm_benchmark.md`) |
| Optimized, well-commented BSM code | Done (`src/models/bsm_chooser.py`) |
| Week 4 evaluation notebook and plots | Done (`notebooks/week4_validation.ipynb`) |

---

## Next Steps (Week 5)

### 1. Two-Approach Architecture

- **Approach 1:** ML volatility prediction (LSTM / RF / XGBoost) + BSM pricing
- **Approach 2:** End-to-end supervised pricing (Linear Regression / GBDT / NN)

### 2. Feature Preparation

- Build strict time-series split datasets: **70% / 15% / 15%** (train / validation / test)
- Ensure no look-ahead bias in feature windows and label construction

### 3. Model Framework Development

- Build initial ML training/evaluation pipelines
- Define common metric interface for fair comparison against Week 4 BSM baseline
- Produce first-round model diagnostics (learning curves, residual patterns, regime-wise errors)

---

## References

- Week 4 validation deliverable: `docs/week4_validation_report.md`
- Week 4 benchmark: `docs/bsm_benchmark.md`
- Week 3 report: `docs/weekly_reports/week3_report.md`
- Project outline: Avalok Capital internship plan

