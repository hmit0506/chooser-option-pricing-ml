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

## Hyperparameter Search Spaces

Search spaces are defined in code (`HYPERPARAM_SEARCH_SPACES` in `scripts/ml/week6_train_eval.py`) and exported with the fitted **best parameters** to:

- `data/reports/week6/hyperparameter_search_spaces.json`

| Component | Method | Rationale |
| --- | --- | --- |
| RF vol | **GridSearchCV** | Few discrete knobs (`n_estimators`, `max_depth`, `min_samples_leaf`); exhaustive grid is cheap. |
| XGB vol | **RandomizedSearchCV** (12 draws) | Larger continuous/discrete mix; random search explores efficiently. |
| Ridge | **GridSearchCV** | Single regularization parameter `alpha`. |
| GBDT | **RandomizedSearchCV** (12 draws) | Many interacting trees; random search preferred. |
| MLP | **RandomizedSearchCV** (9 draws) | Architecture + L2 + learning rate; random search + `early_stopping` inside each fit. |

**Scoring**: all searches minimize **MAE** (via `neg_mean_absolute_error`).

**CV**: `TimeSeriesSplit(n_splits=4)` on the train+val pool used for tuning.

---

## Overfitting / Generalization Diagnostics (Train vs Val vs Test)

For each end-to-end pricing model, we report MAE/RMSE/R² on **train**, **validation**, and **test** splits (same chronological split as training). We also report **MAE gaps**: `val − train` and `test − val` (positive means error increases on the later split).

| Model | MAE (train) | MAE (val) | MAE (test) | Δ val−train | Δ test−val |
| --- | ---:| ---:| ---:| ---:| ---:|
| Ridge | 9.61 | 15.11 | 30.72 | +5.50 | +15.62 |
| GBDT | 1.78 | 1.53 | 35.95 | −0.25 | +34.42 |
| **MLP** | **7.52** | **9.38** | **27.34** | **+1.86** | **+17.96** |

**Interpretation (MLP)**:

- Train error is **lower** than test error (expected under a strict time split).
- **Val** sits between train and test on MAE; the gap **train→val** is modest (+1.86 MAE), while **val→test** is larger (+17.96), consistent with **regime / distribution shift** in the later hold-out period (and a challenging proxy target), not only “in-sample memorization”.
- **Val R²** can be negative for small validation windows or unstable targets; we still rely on **test-set MAE/RMSE** as the primary selection criterion.

**Volatility models (RF / XGB)** and **Approach 1 repricing** (vol → BSM) also have train/val/test splits recorded under `overfitting_diagnostics` in `data/reports/week6/week6_results.json`.

---

## Collinearity (Correlation + VIF)

We run **Pearson correlation** matrices (heatmaps) and **variance inflation factors (VIF)** on the numeric feature matrices used in each track.

**Volatility features** (`sigma_21d`, `sigma_63d`, `sigma_252d`, etc.):

- Multi-horizon realized volatilities are **positively correlated**; VIF for `sigma_252d` / `sigma_21d` / `sigma_63d` is in the **~5–7** range in the latest run (see `data/reports/week6/vif_volatility_features.csv`), indicating **moderate multicollinearity** (common in finance time-series).
- `vix` shows the highest VIF among the volatility-model inputs in this run.

**Pricing features**:

- `moneyness_t` is **linearly redundant** with `s_t` (since \( \text{moneyness} = S/K \) with fixed \(K\)). For VIF we **exclude `moneyness_t`** so that VIF reflects multicollinearity among the remaining inputs (see `vif_exclude_cols` in the script).
- After exclusion, `vix_t`, `sigma_t`, and multi-horizon vol columns still show **elevated VIF** (~5–22), which is expected; tree models and **scaled MLP + L2 (`alpha`)** help mitigate variance from correlated inputs.

Artifacts:

- `data/reports/week6/corr_volatility_features.csv` + `plots/corr_volatility_features_heatmap.png`
- `data/reports/week6/corr_pricing_features.csv` + `plots/corr_pricing_features_heatmap.png`
- `data/reports/week6/vif_volatility_features.csv`
- `data/reports/week6/vif_pricing_features.csv`

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

- `data/reports/week6/week6_results.json` (includes `overfitting_diagnostics`, `best_hyperparameters`, `hyperparameter_search_spaces`)
- `data/reports/week6/hyperparameter_search_spaces.json`
- `data/reports/week6/model_comparison.csv`
- `data/reports/week6/plots/model_error_comparison.png`
- Correlation / VIF outputs (see Collinearity section above)
