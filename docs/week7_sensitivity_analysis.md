# Week 7 Extended Sensitivity Analysis

## Scope

This report covers:

1. **SHAP-based impact quantification** for pricing-model inputs, with emphasis on **VIX level** (`vix_t`) and **sentiment proxy** (`sentiment_proxy`).
2. **Extreme scenario testing** on the **Rubinstein closed-form chooser**:
   - **+50% volatility shock**: multiply annualized \(\sigma\) by **1.5**.
   - **+200 bps rate shock**: add **0.02** to the annual risk-free rate \(r\).
   - **Combined** shocks.

Implementation: `scripts/analysis/week7_sensitivity.py`  
Artifacts: `data/reports/week7/`

---

## 1. SHAP setup (interpretable surrogate)

End-to-end **MLP** (Week 6) is not a tree; for **consistent, fast SHAP** we follow the same pattern as Week 6 interpretability and train a **gradient boosted tree regressor** on the **same pricing feature matrix** (train+val), then compute **Tree SHAP** on a **test subsample**.

| Setting | Value |
| --- | --- |
| Model | `GradientBoostingRegressor` (300 trees, `max_depth=3`, `lr=0.05`) |
| SHAP explainer | `TreeExplainer` with background sample from train+val |
| Explain set | Random subset of test rows (seed 42, cap 800 rows) |
| Label | Proxy realized PV (`direct` target from `build_pricing_dataset`) |

**Note:** Mean absolute SHAP ranks **local contributions** on this surrogate; they should be read as **relative importance** among inputs, not as literal dollar Greeks.

### 1.1 Mean |SHAP| ranking (latest run)

Full table: `data/reports/week7/shap_mean_abs_impact.csv`

**Highlighted regime features (teacher focus):**

| Feature | Mean \|SHAP\| (latest run) |
| --- | ---: |
| `sentiment_proxy` | 0.714 |
| `vix_t` | 0.440 |

In this run, **sentiment_proxy** had a slightly larger mean |SHAP| than **vix_t**, while **rates and volatility levels** (`r_t`, `sigma_t`) and **BSM price** still dominated the top of the global ranking—consistent with the pricing target being strongly tied to level/moneyness and discounting.

Plots:

- `data/reports/week7/plots/shap_summary_week7.png`
- `data/reports/week7/plots/shap_mean_abs_bar.png`

---

## 2. Extreme scenario testing (BSM chooser)

We stress the **Rubinstein chooser** formula using:

- **Baseline**: parameters from the **last row** of the engineered panel (spot, realized \(\sigma_{252}\), row \(r\)) and from **`config/model_params.yaml`** (paper-style parameters) — two CSVs are produced.

### 2.1 Last row of panel (illustrative)

File: `data/reports/week7/extreme_scenarios_last_row.csv`

| Scenario | Chooser price | vs baseline |
| --- | ---:| ---:|
| Baseline | 84.99 | 0% |
| \(\sigma \times 1.5\) | 87.82 | +3.33% |
| \(r + 0.02\) | 87.75 | +3.24% |
| Both | 90.22 | +6.15% |

### 2.2 Config / paper parameters

File: `data/reports/week7/extreme_scenarios_config_params.csv`

Use the same shock definitions; levels differ because spot, \(\sigma\), and \(r\) come from `model_params.yaml`.

---

## 3. Reproducibility

```bash
venv/bin/python scripts/analysis/week7_sensitivity.py
```

Summary JSON: `data/reports/week7/week7_sensitivity_summary.json`

---

## 4. Limitations

- SHAP values are computed on a **tree surrogate**, not the final MLP, to enable exact Tree SHAP.
- Extreme scenarios are **deterministic** shifts to \((r,\sigma)\); they do not re-estimate dividend yield or correlation structure.
- “Sentiment” remains a **VIX-derived proxy**, not news-based sentiment.

---

## 5. Teacher Follow-up: Segmented SHAP

To address whether `vix_t` and `sentiment_proxy` contributions differ by product conditions, we added two segmented analyses:

1. **By maturity bucket** (`T2=0.5y / 1.0y / 1.5y`, with `T1=0.5*T2`)
2. **By moneyness bucket** (`OTM`, `ATM`, `ITM` by `moneyness_t`)

Artifacts:

- `data/reports/week7/shap_by_maturity_bucket.csv`
- `data/reports/week7/shap_by_moneyness_bucket.csv`

These tables report mean absolute SHAP values for `vix_t` and `sentiment_proxy` in each bucket.

---

## 6. Teacher Follow-up: Historical Extreme-event Calibration

To validate stress settings against real market history, we added event-window calibration for:

- **2020-03 crash period** (COVID risk-off)
- **2022 hiking cycle** (aggressive Fed tightening)

For each window, we compare observed extremes with the synthetic stress parameters:

- observed \(\sigma_{252}\) peak / median vs scenario \(1.5\times\)
- observed \(r\) peak - median vs scenario \(+0.02\)
- observed VIX peak / median (context)

Artifact:

- `data/reports/week7/historical_event_calibration.csv`

This gives a direct sanity check on whether `+50% vol` and `+200 bps` are conservative or insufficient relative to historical windows.
