# Week 7 Report — Advanced Analysis & Tool Development

## Completed

1. **Extended sensitivity analysis**
   - Script `scripts/analysis/week7_sensitivity.py`: SHAP (tree surrogate) with emphasis on `vix_t` and `sentiment_proxy`; extreme Rubinstein scenarios (+50% vol as \(\sigma\times1.5\), +200 bps on \(r\), combined).
   - Follow-up added: segmented SHAP by maturity/moneyness and historical calibration against 2020/2022 event windows.
   - Artifacts under `data/reports/week7/` and narrative in `docs/week7_sensitivity_analysis.md`.

2. **Tool framework**
   - **Streamlit** prototype: `app/streamlit_app.py` — parameter inputs, Rubinstein price, quick stress panel, buttons to refresh Yahoo raw data and show latest quotes.
   - **FastAPI** skeleton: `app/api/main.py` — `/health`, `/price/rubinstein`, `/data/update_market`, `/data/latest_quotes`, `/config/defaults`.

3. **Real-time data integration**
   - Module `src/data/market_updater.py` — near-real-time Yahoo download, merge into existing Parquet/CSV, quote snapshot helper.

4. **Planning doc**
   - `docs/week7_project_timeline.md` — core modules table and suggested commands.

## Dependencies

- Added `fastapi` and `uvicorn[standard]` to `requirements.txt`.

## Next steps (Week 8+)

- Complete dual-pricing tool with error margin dashboard.
- Package final report, demo recording, and presentation deck.
- Optional: deployment hardening (Docker/auth/logging) and scheduled refresh.
