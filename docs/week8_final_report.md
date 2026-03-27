# Final Project Report (Week 8)

## 1. Executive Summary

This project builds an end-to-end chooser-option research and tooling pipeline:

1. Data acquisition and preprocessing with real market signals (JPM, VIX, rates).
2. BSM/Rubinstein baseline replication and regime diagnostics.
3. ML enhancement with two architectures and time-series evaluation.
4. Sensitivity, interpretability, and deployable tool prototype (Streamlit + FastAPI).

Core result: the best end-to-end ML model outperforms the BSM baseline on proxy target error metrics, while sensitivity and stress testing clarify where the model is stable and where uncertainty remains high.

---

## 2. Problem Setup

A simple chooser option lets the holder decide at \(T_1\) whether the contract becomes a European call or put (same \(K, T_2\)). The project combines:

- Baseline finance model (Rubinstein/BSM),
- Real-world market features,
- Machine-learning refinements for improved pricing accuracy.

Because public chooser transaction data is sparse, validation uses a reproducible realized-proxy target documented in Week 4.

---

## 3. Data and Feature Pipeline

### 3.1 Data sources

- Yahoo Finance: JPM OHLCV, VIX, dividends
- FRED: Treasury yields (with fallback handling if key/data unavailable)

### 3.2 Engineering pipeline

- Missing value interpolation
- IQR outlier handling
- Trading-day alignment and forward-fill where appropriate
- Feature set including returns, multi-window vol, VIX/rate momentum, sentiment proxy

Main scripts/modules:

- `scripts/data_collection/collect_all.py`
- `src/preprocess.py`
- `src/features/feature_engineering.py`

---

## 4. Modeling Stages

### 4.1 Week 3–4 Baseline

- Monte Carlo chooser + Rubinstein closed-form baseline
- Error analysis and regime split by volatility/sentiment

### 4.2 Week 5–6 ML expansion

Two tracks:

1. **Approach 1:** volatility model \(\rightarrow\) BSM repricing
2. **Approach 2:** end-to-end supervised pricing

Training/evaluation standards:

- strict chronological split (70/15/15)
- time-series CV for tuning
- MAE/RMSE/R² as primary metrics
- SHAP/LIME interpretability outputs

---

## 5. Key Results

### 5.1 Performance

Week 6 benchmark outputs are stored in:

- `data/reports/week6/model_comparison.csv`
- `docs/week6_comparative_analysis.md`

Summary: best ML model improves MAE/RMSE versus BSM baseline on test set.

### 5.2 Robustness diagnostics

- Overfitting diagnostics (train/val/test + gap metrics)
- Search-space + best-parameter export
- Collinearity checks (correlation + VIF)

Artifacts:

- `data/reports/week6/week6_results.json`
- `data/reports/week6/hyperparameter_search_spaces.json`
- `data/reports/week6/vif_*.csv`

---

## 6. Extended Sensitivity and Stress Tests (Week 7)

### 6.1 SHAP impact quantification

- Global SHAP ranking
- `vix_t` and `sentiment_proxy` highlighted explicitly
- Added segmented SHAP by:
  - maturity bucket
  - moneyness bucket

### 6.2 Extreme scenarios and historical calibration

Scenarios:

- +50% volatility
- +200 bps rates
- combined

Calibration against event windows:

- 2020 crash
- 2022 hike cycle

Artifacts:

- `data/reports/week7/shap_by_maturity_bucket.csv`
- `data/reports/week7/shap_by_moneyness_bucket.csv`
- `data/reports/week7/historical_event_calibration.csv`

---

## 7. Tool Completion (Week 8)

### 7.1 Deployable components

- **Streamlit app**: `app/streamlit_app.py`
  - dual pricing (BSM + best ML)
  - error margin display
  - dashboard for trend/errors/metrics/sensitivity tables
- **FastAPI service**: `app/api/main.py`
  - `/price/rubinstein`
  - `/price/dual`
  - `/dashboard/series`
  - `/dashboard/metrics`
  - `/dashboard/sensitivity`
  - `/data/update_market`, `/data/latest_quotes`

### 7.2 Real-time integration

- `src/data/market_updater.py` for incremental JPM/VIX refresh and merge

---

## 8. Limitations

1. Target is a proxy rather than fully observed chooser transactions.
2. Sentiment remains proxy-based (VIX-derived), not NLP-news sentiment.
3. Stress scenarios are deterministic shocks; full regime simulation remains future work.

---

## 9. Practical Impact

- Delivers a reproducible quant workflow from raw data to API/UI tool.
- Provides baseline-vs-ML evidence and explainability artifacts.
- Enables future extension to production serving and richer data feeds.

---

## 10. Next Steps

1. Integrate richer sentiment sources (news/NLP).
2. Add walk-forward retraining schedule.
3. Add deployment hardening (auth, logging, model versioning, containerization).
4. Validate against additional market proxies/instruments where available.

