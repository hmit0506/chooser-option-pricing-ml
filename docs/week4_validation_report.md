# Week 4 Validation Report – BSM Baseline Performance Evaluation

**Project:** Avalok Capital Quantitative Research & Trading - Advanced Chooser Option Pricing Model with Real-World Data & Machine Learning  
**Author:** Shi Qian  
**Date:** February 2026  
**Week:** 4  
**Version:** 1.1  

---

## Executive Summary

This report evaluates the Week 3 BSM chooser model and formalizes a baseline for subsequent ML benchmarking.  
Because direct public CME chooser transaction prices are not available in the current data pipeline, validation is conducted using a transparent and reproducible **historical realized proxy** framework.

Backtest results indicate that BSM provides a stable directional baseline but exhibits substantial pricing error under real market dynamics, supporting the need for regime-aware ML extensions in Weeks 5–6.

---

## 1. Validation Objective

The Week 4 objective is to:
1. Quantify baseline pricing accuracy with MAE and RMSE.
2. Diagnose failure patterns across volatility/sentiment regimes.
3. Produce benchmark artifacts reusable for ML model comparison.

---

## 2. Data and Methodology

### 2.1 Data Inputs
- JPM historical daily prices: `data/raw/yahoo_finance/JPM_daily_ohlcv.parquet`
- VIX daily prices: `data/raw/yahoo_finance/VIX_daily.parquet`
- 10Y Treasury (DGS10): `data/raw/fred/DGS10.parquet`
- Model parameters: `config/model_params.yaml`

### 2.2 Practical Data Constraint
Public direct CME chooser transaction series is not readily accessible in this project environment.  
Accordingly, the analysis uses an internal historical proxy target while preserving strict out-of-sample logic.

### 2.3 Proxy-Actual Construction
For each valuation date `t`:
1. Compute predicted chooser value at `t` using Rubinstein closed-form (`pred_price`).
2. Observe realized JPM prices at:
   - `t + T1` (decision date),
   - `t + T2` (maturity date),
   with `T1 ≈ 126` and `T2 ≈ 252` trading days.
3. Compute realized chooser payoff using the paper rule (`S_T1 > K` -> call; otherwise put).
4. Discount payoff to `t` to obtain `actual_proxy_pv`.
5. Compute error metrics between `pred_price` and `actual_proxy_pv`.

---

## 3. Backtest Scope

- Sample size: **1308** observations  
- Date range: **2018-10-17 to 2023-12-28**  
- Volatility estimate: rolling 252-day realized sigma  
- Regime threshold: VIX > 30 classified as `high_vol`

---

## 4. Quantitative Results

### 4.1 Overall Metrics

| Metric | Value |
|---|---:|
| MAE | 22.6089 |
| RMSE | 26.6352 |
| MAPE | 353.60% |

**Interpretation:**  
MAE/RMSE are used as primary benchmark metrics. MAPE is reported for completeness but is unstable when proxy-actual values approach zero.

### 4.2 Regime Metrics (VIX Split)

| Regime | Count | MAE | RMSE | MAPE |
|---|---:|---:|---:|---:|
| high_vol | 137 | 22.8178 | 25.5121 | 2.4695 |
| normal_vol | 1171 | 22.5845 | 26.7635 | 3.6755 |

### 4.3 Sentiment-Proxy Split

Sentiment proxy is inverse-normalized VIX (rolling min-max).

| Regime | Count | MAE | RMSE | MAPE |
|---|---:|---:|---:|---:|
| low_sentiment | 44 | 23.5959 | 28.2986 | 1.8018 |
| normal_sentiment | 1264 | 22.5746 | 26.5754 | 3.6067 |

---

## 5. Diagnostic Visuals

Generated in `notebooks/week4_validation.ipynb`:
- Predicted vs proxy-actual scatter
- Error time series
- 60-day rolling MAE
- MAE by VIX regime

Artifact:
- `notebooks/week4_validation_plots.png`

---

## 6. Limitation Assessment

### 6.1 Regime Dynamics and Volatility Clustering
BSM assumes constant volatility and cannot fully capture clustering, jumps, and abrupt repricing around macro shocks.

### 6.2 Sentiment and Event Channel Omission
The current baseline only includes a simple sentiment proxy; event-driven information is not explicitly modeled, which contributes to residual errors.

### 6.3 Target Data Representativeness
The proxy-actual target is a necessary engineering substitute, not a direct CME chooser transaction series.  
This should be explicitly acknowledged in cross-model comparisons.

---

## 7. Baseline Conclusion

Week 4 delivers a reproducible and transparent BSM benchmark with:
- consistent MAE/RMSE baseline metrics,
- regime-aware diagnostic cuts,
- reusable outputs for Week 5–6 ML benchmarking.

The benchmark confirms BSM as a valid reference model, while highlighting clear headroom for data-driven models under non-stationary market conditions.

