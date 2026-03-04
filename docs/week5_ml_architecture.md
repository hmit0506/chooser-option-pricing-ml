# Week 5 – ML Architecture Design for Chooser Option Pricing

**Project:** Avalok Capital Quantitative Research & Trading - Advanced Chooser Option Pricing Model with Real-World Data & Machine Learning  
**Author:** Shi Qian  
**Week:** 5  
**Version:** 1.0  

---

## 1. Objectives

Week 5 extends the BSM baseline (Weeks 3–4) by:

1. Designing two complementary ML architectures for chooser option pricing.
2. Preparing **time-series-aware** feature datasets (70% / 15% / 15%) without look-ahead bias.
3. Implementing initial ML model frameworks for both approaches.

---

## 2. Two-Approach Architecture

### 2.1 Approach 1 – ML Volatility Forecasting + BSM Pricing

**Idea:**  
Keep the financial structure of BSM, but replace the constant volatility assumption with a **learned, time-varying volatility forecast**.

**Target (label):**

- Future realized volatility over a fixed horizon, e.g.:
  - \( \sigma_{\text{real}}(t, t+T_1) \): realized vol over the next 126 trading days (≈ 0.5 year)
  - Computed from future log returns:
    \[
    \sigma_{\text{real}} = \sqrt{252} \cdot \text{stdev}\big(\log S_{t+i} - \log S_{t+i-1}\big)_{i=1,\dots,H}
    \]

**Features at time \(t\)** (no look-ahead):

- **Price-based:**
  - Past log returns over multi-window lags (e.g. 5/21/63 days)
  - Rolling realized volatility (21d/63d/252d)
  - Moneyness and approximate time-to-maturity
- **Volatility proxies:**
  - VIX level, changes, and short-term VIX volatility
  - Rolling correlation between JPM returns and VIX changes
- **Rates / Macro:**
  - DGS10 level (risk-free rate proxy)
  - Simple rate momentum (e.g. 21-day ΔDGS10)
- **Sentiment proxies:**
  - VIX-based sentiment proxy (1 − normalized VIX) and its rolling stats

**Model families (initial):**

- **Tree-based:**
  - Random Forest Regressor
  - XGBoost Regressor
- **Sequence model (planned):**
  - 1-layer LSTM over sliding windows of returns and VIX (optional extension)

**Pricing step:**

- Use predicted volatility \( \hat{\sigma}(t) \) as input to the Rubinstein chooser formula:
  \[
  \text{BSM\_chooser\_price}(t) = f_{\text{Rubinstein}}\big(S_t, K, r_t, q, \hat{\sigma}(t), T_1, T_2\big)
  \]
- Compare these ML-adjusted BSM prices to the Week 4 proxy-actual baseline.

---

### 2.2 Approach 2 – End-to-End Supervised Pricing

**Idea:**  
Let an ML model learn the mapping from features to **chooser value** (or to the BSM residual) directly, without imposing the BSM functional form.

**Target (label) options:**

1. **Direct pricing target:**
   - \( y(t) = \text{actual\_proxy\_pv}(t) \)  
     (discounted realized payoff proxy from Week 4)
2. **Residual target:**
   - \( y(t) = \text{actual\_proxy\_pv}(t) - \text{BSM\_price}(t) \)  
     i.e. learn only the correction term to BSM.

**Features at time \(t\):**

- Same as Approach 1 (price, vol, VIX, rates, sentiment)
- Optionally include:
  - BSM chooser price as a feature (for residual modeling)
  - Predicted vol \( \hat{\sigma}(t) \) from Approach 1
  - Interactions (e.g. VIX × moneyness, vol × time-to-maturity)

**Model families (initial):**

- Linear Regression / Ridge (as a sanity-check baseline)
- Gradient Boosted Trees (sklearn GBDT or XGBoost)
- Small MLP (1–2 hidden layers, ReLU, early stopping)

**Evaluation:**

- Compare ML predictions vs `actual_proxy_pv(t)` on the test set using:
  - MAE, RMSE, R²
  - Regime-wise metrics (VIX and sentiment splits), consistent with Week 4 baseline

---

## 3. Feature Preparation & Time-Series Splits

### 3.1 Dataset Construction

Both approaches rely on a unified **backtest table** at daily frequency:

- Index: valuation date `t`
- Columns include:
  - `close_t`, `vix_t`, `r_t`, `sigma_252d_t`, sentiment proxy
  - BSM chooser price at `t` (Rubinstein formula)
  - `actual_proxy_pv(t)` (discounted realized payoff from Week 4 pipeline)
  - Future realized vol labels (for Approach 1)

Implementation (Week 5 code):

- `src/ml/datasets.py`
  - `load_base_frame(...)` – constructs the basic aligned DataFrame from raw data.
  - `build_volatility_dataset(...)` – builds (X, y, dates) for volatility forecasting.
  - `build_pricing_dataset(...)` – builds (X, y, dates) for pricing / residual modeling.

### 3.2 Time-Series Split (70% / 15% / 15%)

- Split by **time**, not by random shuffle:
  - Train: earliest 70% of dates
  - Validation: next 15%
  - Test: latest 15%
- Implemented as:
  - `time_series_split(X, y, dates, train_frac=0.7, val_frac=0.15)`

This ensures no look-ahead bias in both training and feature construction.

---

## 4. Model Frameworks (Initial Implementation)

### 4.1 Volatility Models – `src/ml/models_vol.py`

- `train_rf_vol(...)` – Random Forest:
  - Inputs: train/val sets from `build_volatility_dataset`
  - Outputs: fitted model + validation metrics
- `train_xgb_vol(...)` – XGBoost:
  - Similar interface, with basic hyperparameters exposed
- `evaluate_vol_model(...)`:
  - Computes MAE / RMSE / R² on a given split

LSTM-based volatility models are planned as a later extension once tree-based baselines are established.

### 4.2 Pricing Models – `src/ml/models_pricing.py`

- `train_linear_pricing(...)` – Linear / Ridge regression baseline
- `train_gbdt_pricing(...)` – Tree-based non-linear model
- `train_mlp_pricing(...)` – Small feed-forward neural network
- `evaluate_pricing_model(...)` – same metrics as above

All models will be compared against the BSM baseline from Week 4 using the same test set and metrics.

---

## 5. Evaluation & Benchmarking Plan

### 5.1 Metrics

- Primary:
  - MAE
  - RMSE
  - R²
- Regime-wise:
  - VIX regimes (high_vol vs normal_vol)
  - Sentiment regimes (low_sentiment vs normal_sentiment)

### 5.2 Artifacts

- `notebooks/week5_ml_frameworks.ipynb`:
  - Demonstrates:
    - Dataset construction and time-series splits
    - Training 1–2 models per approach
    - Baseline comparison table vs BSM
    - Basic diagnostic plots (pred vs actual, residual histograms)
- Updated `docs/bsm_benchmark.md` (later weeks) to include ML vs BSM comparisons.

---

## 6. Implementation Summary

By the end of Week 5, the codebase will contain:

- **ML architecture design:** `docs/week5_ml_architecture.md`
- **Feature preparation & splits:** `src/ml/datasets.py`
- **Metrics utilities:** `src/ml/metrics.py`
- **Initial model frameworks:** `src/ml/models_vol.py`, `src/ml/models_pricing.py`
- **Demo notebook:** `notebooks/week5_ml_frameworks.ipynb`

These components provide a structured foundation for Weeks 6–7, where hyperparameter tuning, model comparison, and interpretability (e.g. SHAP analysis) will be the main focus.

