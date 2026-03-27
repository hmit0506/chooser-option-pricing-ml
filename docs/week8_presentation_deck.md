# Final Presentation Deck (Week 8)

## Slide 1 — Title
- Advanced Chooser Option Pricing with Real Data + ML
- Internship project summary

## Slide 2 — Motivation
- Chooser pricing under market regime shifts
- Limits of constant-vol BSM in practice

## Slide 3 — System Architecture
- Data layer (Yahoo/FRED)
- Modeling layer (BSM + ML)
- Tooling layer (Streamlit + FastAPI)

## Slide 4 — Data & Features
- Preprocessing pipeline
- Traditional + advanced features

## Slide 5 — Baseline (Week 3–4)
- Rubinstein/BSM replication
- Baseline metrics and failure modes

## Slide 6 — ML Design (Week 5–6)
- Approach 1 vs Approach 2
- Time-series split + CV strategy

## Slide 7 — Main Results
- Test-set table (BSM vs ML)
- Best model choice rationale

## Slide 8 — Robustness Checks
- Train/val/test gap diagnostics
- Hyperparameter search spaces + best params
- Collinearity (corr + VIF)

## Slide 9 — Sensitivity Analysis
- SHAP global ranking
- SHAP by maturity and moneyness buckets
- VIX/sentiment insights

## Slide 10 — Stress and Historical Calibration
- +50% vol, +200 bps, combined
- 2020/2022 event-window calibration

## Slide 11 — Tool Demo
- Dual pricing + error bands
- Trend and performance dashboards
- Real-time data refresh endpoint

## Slide 12 — Conclusion and Next Steps
- What is delivered
- What remains for production hardening

