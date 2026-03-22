# Week 7 — Project Timeline & Core Modules

## Milestone (this week)

| Area | Deliverable | Status |
| --- | --- | --- |
| Sensitivity | SHAP emphasis on VIX + sentiment; extreme \(r,\sigma\) shocks | Report + script + artifacts |
| Tooling | Streamlit UI shell + FastAPI service skeleton | `app/streamlit_app.py`, `app/api/main.py` |
| Data | Auto-update merge for Yahoo raw series | `src/data/market_updater.py` |

---

## Core modules (data integration)

| Module | Responsibility |
| --- | --- |
| `src/data/market_updater.py` | Fetch recent JPM / VIX via `yfinance`, merge into existing Parquet/CSV under `data/raw/yahoo_finance/`, expose `get_latest_quote_summary()` for UI/API. |
| `scripts/data_collection/collect_all.py` | Full historical backfill (existing Week 1–2 path). |
| `src/preprocess.py` | Batch feature pipeline after raw data refresh. |

**Auto-update flow**

1. UI/API calls `update_market_data_raw(lookback_days=...)`.
2. Downstream research runs `python src/preprocess.py` when a full refresh is needed.

---

## Suggested timeline (remaining internship weeks)

| Week | Focus |
| --- | --- |
| 7 | Sensitivity + prototype + live raw merge |
| 8 | Harden API (auth, validation), deploy notes, optional Docker |

---

## Tooling commands

| Component | Command |
| --- | --- |
| Streamlit | `streamlit run app/streamlit_app.py` |
| FastAPI | `uvicorn app.api.main:app --reload` (from repo root) |
| Sensitivity | `venv/bin/python scripts/analysis/week7_sensitivity.py` |
