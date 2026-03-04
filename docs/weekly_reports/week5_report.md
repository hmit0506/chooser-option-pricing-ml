# Week 5 Weekly Report – ML Architecture & Pipeline Setup

**Project:** Avalok Capital Quantitative Research & Trading - Advanced Chooser Option Pricing Model with Real-World Data & Machine Learning  
**Author:** Shi Qian  
**Date:** February 2026  
**Week:** 5  
**Version:** 1.0  

---

## Summary

Week 5 focused on moving from a pure BSM baseline to a **two-approach ML architecture** for chooser option pricing.  
The main outcomes are: a clear ML design document, time-series-aware datasets, and initial ML model frameworks for both volatility-then-BSM and end-to-end pricing approaches.

---

## Completed Work

### 1. ML Architecture Design

- **Location:** `docs/week5_ml_architecture.md`
- Defined two complementary approaches:
  1. **Approach 1 – ML volatility forecasting + BSM pricing**
     - ML models (RF/XGBoost/LSTM planned) forecast future realized volatility.
     - Predicted volatility is plugged into the Rubinstein chooser formula.
  2. **Approach 2 – End-to-end supervised pricing**
     - Linear/GBDT/MLP models map features directly to chooser value or BSM residual.
- Clarified:
  - Feature families (price, VIX, rates, sentiment proxies).
  - Label definitions (future realized vol, proxy-actual PV, residual).
  - Evaluation metrics (MAE, RMSE, R² + regime splits).

### 2. Feature Preparation & Time-Series Splits

- **Location:** `src/ml/datasets.py`
- Implemented:
  - `load_base_frame(...)` – builds aligned JPM/VIX/DGS10 daily frame with:
    - `close`, `vix`, `r`, `log_ret`, `sigma_252d`, `sentiment_proxy`.
  - `build_volatility_dataset(...)` – constructs (X, y, dates) for volatility forecasting using future realized vol labels.
  - `build_pricing_dataset(...)` – constructs (X, y, dates, bsm_price) for pricing / residual modeling using:
    - Rubinstein BSM chooser price.
    - Week 4 `realized_proxy_pv` as target.
  - `time_series_split(...)` – 70%/15%/15% chronological split (train/val/test) to prevent look-ahead bias.

### 3. Initial ML Model Frameworks

- **Location:** `src/ml/`
  - `metrics.py` – common regression metrics (`regression_metrics` for MAE/RMSE/R²).
  - `models_vol.py` – volatility models:
    - `train_rf_vol(...)` – Random Forest baseline.
    - `train_xgb_vol(...)` – XGBoost baseline (optional dependency).
  - `models_pricing.py` – pricing models:
    - `train_linear_pricing(...)` – Linear/Ridge baseline.
    - `train_gbdt_pricing(...)` – Gradient Boosted Trees.
    - `train_mlp_pricing(...)` – small MLP.
  - `__init__.py` – exposes `datasets`, `metrics`, `models_vol`, `models_pricing`.

### 4. Week 5 ML Framework Notebook

- **Location:** `notebooks/week5_ml_frameworks.ipynb`
- Demonstrates:
  - Building volatility and pricing datasets from `load_base_frame(...)`.
  - Applying `time_series_split(...)` for train/val/test.
  - Training:
    - A Random Forest volatility model (Approach 1 demo).
    - A Linear pricing model (Approach 2 demo).
  - Printing basic validation metrics to confirm the pipelines are wired correctly.

---

## Deliverables Checklist

| Deliverable | Status |
|---|---|
| ML architecture design document | Done (`docs/week5_ml_architecture.md`) |
| Feature preparation & optimization code | Done (`src/ml/datasets.py`) |
| Initial ML model frameworks (both approaches) | Done (`src/ml/*.py`, `notebooks/week5_ml_frameworks.ipynb`) |

---

## Next Steps (Week 6)

### 1. Hyperparameter Optimization

- Apply **grid/random search** with time-series–aware validation (e.g., expanding-window or fixed split) for:
  - Volatility models (RF/XGBoost).
  - Pricing models (GBDT/MLP).

### 2. Model Training & Evaluation

- Train final models using the chosen hyperparameters on **train+val** sets.
- Evaluate on the **held-out test set** using MAE, RMSE, and R².

### 3. Performance Comparison vs BSM Baseline

- Compare ML models against the Week 4 BSM baseline across:
  - Overall MAE/RMSE/R².
  - Regime-wise metrics (VIX and sentiment splits).
- Summarize improvements (absolute and percentage) for each approach.

### 4. Interpretability Analysis

- Integrate SHAP / LIME for selected models (e.g., XGBoost / GBDT / MLP) to:
  - Identify key drivers (VIX dynamics, realized volatility, rate level, etc.).
  - Confirm whether ML models are learning economically sensible patterns.

---

## References

- BSM baseline & error analysis: `docs/week4_validation_report.md`, `docs/bsm_benchmark.md`
- ML architecture design: `docs/week5_ml_architecture.md`
- Weekly progress: `docs/weekly_reports/week2_report.md`, `week3_report.md`, `week4_report.md`

