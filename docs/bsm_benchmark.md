# BSM Benchmark Baseline (Week 4)

This document defines the official BSM baseline used for ML comparison in Weeks 5–6.

---

## 1. Baseline Definition

### Model
- Chooser option pricing under BSM framework
- Primary predictor: Rubinstein (1991) closed-form chooser value
- Reference implementation: `src/models/bsm_chooser.py`

### Input Parameters
- Strike `K = 150`
- Decision time `T1 = 0.5`
- Maturity `T2 = 1.0`
- Dividend yield `q = 0.0233`
- Volatility estimator: rolling 252-day realized volatility
- Risk-free rate: DGS10 (fallback to config value when missing)

### Validation Target
- `actual_proxy_pv` = discounted realized chooser payoff derived from historical JPM path:
  - choose call if `S_T1 > K`, else put,
  - payoff at `T2`, discounted to `t`.

---

## 2. Baseline Performance Snapshot

| Metric | Value |
|---|---:|
| Sample count | 1308 |
| MAE | 22.6089 |
| RMSE | 26.6352 |
| MAPE | 353.60% |

Backtest range: 2018-10-17 to 2023-12-28

---

## 3. Regime Diagnostics

### VIX Regime (`threshold = 30`)

| Regime | Count | MAE | RMSE |
|---|---:|---:|---:|
| high_vol | 137 | 22.8178 | 25.5121 |
| normal_vol | 1171 | 22.5845 | 26.7635 |

### Sentiment Proxy Regime

| Regime | Count | MAE | RMSE |
|---|---:|---:|---:|
| low_sentiment | 44 | 23.5959 | 28.2986 |
| normal_sentiment | 1264 | 22.5746 | 26.5754 |

---

## 4. Benchmark Artifacts

- Validation notebook: `notebooks/week4_validation.ipynb`
- Plots: `notebooks/week4_validation_plots.png`
- Metric exports:
  - `docs/week4_metrics_summary.csv`
  - `docs/week4_regime_metrics.csv`
- Full report: `docs/week4_validation_report.md`

---

## 5. Use in Week 5–6

All candidate ML models should be evaluated on the same sample and compared against this baseline using:
- MAE improvement (%)
- RMSE improvement (%)
- Regime-specific error improvements (high-vol and low-sentiment subsets)

This ensures fair and transparent model comparison beyond aggregate metrics.

