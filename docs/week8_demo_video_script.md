# Demo Video Script (5–10 minutes)

## 0:00–0:45 — Project Intro
- State objective: improve chooser pricing with data + ML.
- Mention final deliverables: tool + report + reproducible artifacts.

## 0:45–2:00 — Data & Pipeline
- Show folder structure and data flow:
  - raw collection
  - preprocessing
  - processed features
- Mention robustness handling (missing FRED fallback).

## 2:00–3:15 — Modeling Summary
- Briefly compare:
  - BSM baseline
  - ML approaches
- Show key benchmark table from Week 6.

## 3:15–4:30 — Explainability & Sensitivity
- Show SHAP summary plot.
- Highlight VIX/sentiment segmented tables.
- Show stress scenarios and historical calibration CSV.

## 4:30–7:30 — Live Tool Demo
- Run Streamlit app:
  - input parameters
  - dual price output (BSM vs ML)
  - error margin display
  - dashboard charts (trend + error)
  - sensitivity/performance tables
- Trigger market refresh button and show returned status.

## 7:30–8:30 — API Demo
- Call `/price/dual`, `/dashboard/metrics`, `/data/latest_quotes`.
- Show JSON outputs quickly.

## 8:30–9:30 — Closing
- Summarize gains and current limitations.
- Next steps: deployment hardening, richer sentiment, retraining loop.

