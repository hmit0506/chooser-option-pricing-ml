# Week 6 Comparative Analysis (ML vs BSM)

## Objective

Week 6 focuses on:

1. Hyperparameter optimization with time-series-aware cross-validation.
2. Final model training and test-set evaluation.
3. Benchmark comparison against the Week 4 BSM baseline.
4. Interpretability outputs using SHAP and LIME.

---

## Experimental Setup

- **Data split**: chronological `70% / 15% / 15%` (train/val/test).
- **Pricing target**: historical realized proxy PV for chooser value.
- **Baseline**: Rubinstein closed-form chooser price (`bsm_price`).
- **Approach 1**: volatility forecast then BSM repricing.
  - Vol models: Random Forest (grid search), XGBoost (randomized search)
- **Approach 2**: end-to-end supervised pricing.
  - Models: Ridge, GBDT, MLP (grid/randomized search)

All hyperparameter searches use **time-series cross-validation** (`TimeSeriesSplit`) to reduce look-ahead bias.

---

## Test-Set Performance Summary

| Model | MAE | RMSE | R² | MAE Improvement vs BSM | RMSE Improvement vs BSM |
|---|---:|---:|---:|---:|---:|
| BSM baseline | 34.7897 | 37.8863 | -0.5966 | - | - |
| Approach1 RF-vol + BSM | 33.7566 | 36.1401 | -0.4528 | +2.97% | +4.61% |
| Approach1 XGB-vol + BSM | 32.5829 | 35.0129 | -0.3636 | +6.34% | +7.58% |
| Approach2 Ridge | 30.7227 | 33.5475 | -0.2519 | +11.69% | +11.45% |
| Approach2 GBDT | 35.9517 | 43.3389 | -1.0892 | -3.34% | -14.39% |
| **Approach2 MLP** | **27.3385** | **29.8961** | **0.0058** | **+21.42%** | **+21.09%** |

**Primary outcome**: the best end-to-end model is **MLP**, which outperforms the BSM baseline and all other tested models on MAE/RMSE.

---

## Model Selection Results

- **Best Approach 1 vol model**: `xgb_vol`
- **Best Approach 2 pricing model**: `mlp`
- **Selected primary model for final pricing**: `approach2_mlp` (lowest test MAE/RMSE)

Saved model artifacts:

- `models/week6/best_vol_model_xgb_vol.pkl`
- `models/week6/best_pricing_model_mlp.pkl`

---

## Interpretability Outputs

Generated files:

- SHAP summary plot: `data/reports/week6/plots/shap_summary.png`
- SHAP bar plot: `data/reports/week6/plots/shap_bar.png`
- LIME local explanation (sample): `data/reports/week6/lime_explanation_sample0.html`

Interpretability uses the tree-based pricing model (`gbdt`) for stable SHAP TreeExplainer output, while final model selection remains based on test performance (MLP).

---

## Key Takeaways

1. ML models provide measurable gains over BSM under the current proxy target.
2. Volatility-prediction enhancement improves BSM, but end-to-end MLP achieves the largest gain.
3. Not all nonlinear models are superior: GBDT underperforms here, highlighting the importance of model-specific tuning and validation.

---

## Reproducibility

Run the Week 6 pipeline from project root:

```bash
venv/bin/python scripts/ml/week6_train_eval.py
```

Generated summary artifacts:

- `data/reports/week6/week6_results.json`
- `data/reports/week6/model_comparison.csv`
- `data/reports/week6/plots/model_error_comparison.png`
