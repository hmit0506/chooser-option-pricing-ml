# Week 3 Weekly Report – Original BSM Model Replication

**Project:** Avalok Capital Quantitative Research & Trading - Advanced Chooser Option Pricing Model with Real-World Data & Machine Learning  
**Author:** Shi Qian  
**Date:** February 18, 2026  
**Week:** 3  
**Version:** 1.0  

---

## Summary

Week 3 focused on replicating the BSM-based chooser option pricing model from the reference paper (Huang, Wang & Wan, 2021). The model was implemented via both Monte Carlo simulation and the Rubinstein (1991) closed-form formula. Validation against the paper's Table 3 and sensitivity analysis confirmed correctness. The BSM baseline is now established for ML comparison in later weeks.

---

## Completed Work

### 1. Parameter Configuration

- **Location:** `config/model_params.yaml`
- All parameters from the paper's Table 2:

| Parameter | Value | Description |
|-----------|-------|-------------|
| $S_0$ | $156.70 | JPM stock price (23 Aug 2021) |
| $K$ | $150.00 | CME option strike price |
| $r$ | 0.15% | Risk-free rate |
| $\sigma$ | 28.2% | Historical volatility (Aug 2020–2021) |
| $q$ | 2.33% | Dividend yield |
| $T_1$ | 0.5 years | Decision date |
| $T_2$ | 1.0 year | Maturity date |

### 2. BSM Chooser Pricing Module

- **Location:** `src/models/bsm_chooser.py`
- Modular functions with no external state:
  - `simulate_gbm_paths()` — Risk-neutral GBM terminal prices
  - `chooser_payoffs()` — Paper's simplified rule ($S_{T_1} > K$)
  - `chooser_payoffs_proper()` — Proper BSM-value comparison at $T_1$
  - `price_chooser_mc()` — Full 2-period Monte Carlo pricer with discounting
  - `bsm_call()` / `bsm_put()` — Standard BSM with continuous dividends
  - `rubinstein_chooser()` — Rubinstein (1991) closed-form analytic price

### 3. BSM Pricing Notebook

- **Location:** `notebooks/week3_bsm_pricing.ipynb`
- Demonstrates GBM simulation, chooser decision logic, and Monte Carlo pricing
- Key results (paper parameters, N=10,000 paths, seed=42):
  - **MC chooser price:** $28.97 (SE: $0.31)
  - **Rubinstein analytic:** $29.13
  - **Relative error:** 0.55%
  - **Call ratio at $T_1$:** 52.4%
- Confirms pricing hierarchy: BSM Call ($18.69) < Chooser ($29.13) < Straddle ($34.06)

### 4. Model Validation Notebook

- **Location:** `notebooks/week3_validation.ipynb`
- **Table 3 reproduction:**
  - Paper's choice logic ($S_{T_1} > K$ → call, else put) verified exactly for all 10 paths
  - Payoff formulas confirmed: $\max(S_{T_2} - K, 0)$ for call, $\max(K - S_{T_2}, 0)$ for put
  - Cross-validated by running 1,000 simulations per paper $S_{T_1}$ value
- **Sensitivity analysis** (Figures 3–6 from paper reproduced):
  - Higher $\sigma$ → higher call, put, and chooser values
  - Higher $K$ → lower call, higher put
  - Higher $r$ → higher call, lower put
  - Higher $q$ → lower call, higher put
- **MC convergence:** Standard error decays as $1/\sqrt{N}$; 10,000+ paths sufficient for SE < $0.50

### 5. Dependencies

- Added `pyyaml>=6.0` to `requirements.txt` for YAML config loading

---

## Deliverables Checklist

| Deliverable | Status |
|-------------|--------|
| Fully functional BSM model code (notebook) | Done |
| Model validation with paper comparison | Done |
| Parameter configuration file (YAML) | Done |
| Reusable pricing module (`src/models/`) | Done |
| Sensitivity analysis plots | Done |

---

## Next Steps (Week 4)

### Performance Evaluation & Limitation Analysis

1. **Error metric calculation:** Compute MAE and RMSE between BSM model predictions and actual CME transaction prices to quantify pricing accuracy.
2. **Limitation analysis:** Identify failure modes in high-volatility periods (e.g., COVID-19 March 2020) and sentiment impact gaps where the constant-volatility BSM assumption breaks down.
3. **Benchmark establishment:** Document BSM model performance baseline (pricing errors, volatility regimes, call/put ratio accuracy) for comparison with ML models in Weeks 5–6.

---

## References

- Huang, Z., Wang, X., & Wan, W. (2021). Exploration of JPMorgan Chooser Option Pricing. *BCP Business & Management*, 15, 93–102.
- Rubinstein, M. (1991). Options for the Undecided. *Risk*, 4(4).
- Week 2: `docs/weekly_reports/week2_report.md`
- Feature details: `docs/feature_engineering.md`
- Project Outline: Avalok Capital PDF
