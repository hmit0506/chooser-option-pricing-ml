# Week 8 Report — Tool Completion and Final Packaging

## Completed

1. **Tool feature completion**
   - Implemented dual pricing (BSM/Rubinstein + best ML model).
   - Added error margin display from historical residuals.
   - Added dashboard panels for:
     - price trends (proxy actual vs BSM vs ML),
     - rolling absolute errors,
     - performance metrics table,
     - sensitivity tables (maturity/moneyness/event calibration).

2. **Teacher follow-up implementation**
   - Added SHAP segmented analysis by maturity/moneyness.
   - Added historical extreme-event calibration against 2020 crash and 2022 hiking windows.

3. **Final materials preparation**
   - Final report draft: `docs/week8_final_report.md`
   - Demo video script: `docs/week8_demo_video_script.md`
   - Presentation deck outline: `docs/week8_presentation_deck.md`

## Core Files Added/Updated

- `src/tooling/pricing_tool.py`
- `app/streamlit_app.py`
- `app/api/main.py`
- `scripts/analysis/week7_sensitivity.py` (extended)
- `docs/week8_final_report.md`
- `docs/week8_demo_video_script.md`
- `docs/week8_presentation_deck.md`

## Deliverable Status

| Deliverable | Status |
| --- | --- |
| Fully deployable pricing tool (repo + README) | Completed in repository structure and run commands |
| Final project report | Draft completed (`docs/week8_final_report.md`) |
| Tool demo video (5–10 min) | Script prepared (`docs/week8_demo_video_script.md`) |
| Final presentation deck | Slide outline prepared (`docs/week8_presentation_deck.md`) |

