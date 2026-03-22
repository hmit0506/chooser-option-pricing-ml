# Week 6 Report: Hyperparameter Optimization, Final Evaluation, and Interpretability

## Period Summary

This week completed the Week 6 milestone by running time-series-aware hyperparameter tuning, training final candidate models, benchmarking ML methods against the BSM baseline, and generating SHAP/LIME interpretability artifacts.

---

## Completed Work

### 1. Hyperparameter Optimization

- Implemented a dedicated Week 6 pipeline script: `scripts/ml/week6_train_eval.py`.
- Applied **time-series cross-validation** (`TimeSeriesSplit`) to reduce leakage risk.
- Used:
  - Grid search for Random Forest volatility model and Ridge pricing model.
  - Randomized search for XGBoost volatility model, GBDT pricing model, and MLP pricing model.

### 2. Final Training & Test Evaluation

- Trained and evaluated both architecture tracks:
  - **Approach 1**: volatility forecasting + BSM repricing
  - **Approach 2**: end-to-end supervised pricing
- Test-set metrics (MAE/RMSE/R²) were produced and saved to:
  - `data/reports/week6/week6_results.json`
  - `data/reports/week6/model_comparison.csv`

### 3. ML vs BSM Performance Comparison

- Established direct benchmark comparison against BSM baseline.
- Best model this week: **Approach 2 MLP**.
- Saved final model artifacts:
  - `models/week6/best_vol_model_xgb_vol.pkl`
  - `models/week6/best_pricing_model_mlp.pkl`

### 4. Interpretability (SHAP/LIME)

- Generated SHAP and LIME artifacts:
  - `data/reports/week6/plots/shap_summary.png`
  - `data/reports/week6/plots/shap_bar.png`
  - `data/reports/week6/lime_explanation_sample0.html`

### 5. Follow-up diagnostics (teacher Q&A)

- **Train / val / test error tables** for end-to-end pricing models (Ridge, GBDT, MLP) and MAE generalization gaps.
- **Documented hyperparameter grids** + exported **best parameters** to `data/reports/week6/hyperparameter_search_spaces.json`.
- **Collinearity checks**: Pearson correlation heatmaps + VIF tables for volatility and pricing feature matrices (pricing VIF excludes redundant `moneyness_t` vs `s_t`).

---

## Deliverables Status

| Week 6 Deliverable | Status |
|---|---|
| Trained ML models (pickle files) | Done |
| Comparative analysis report with metrics | Done (`docs/week6_comparative_analysis.md`) |
| Feature importance visualizations (SHAP/LIME outputs) | Done |

---

## Next Steps (Week 7)

1. Package the selected model pipeline into a deployment-oriented interface.
2. Add model-serving wrappers and consistent inference inputs.
3. Build initial Streamlit/FastAPI integration for pricing demo and validation.
4. Add production-style checks (input validation, inference logging, reproducibility config).
