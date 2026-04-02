# Advanced Chooser Option Pricing Model with Machine Learning

## 📋 Project Overview
This project is part of the Avalok Capital Quantitative Research Internship, focused on developing an enhanced pricing model for Chooser Options using both traditional Black-Scholes-Merton (BSM) framework and modern Machine Learning techniques.

**Project Duration**: 8 Weeks   
**Organization**: Avalok Capital  

## 🎯 Project Objectives
1. Replicate and validate the BSM-based Chooser Option pricing model
2. Develop ML-enhanced pricing models (LSTM, XGBoost, Neural Networks)
3. Build a production-ready pricing tool with real-time capabilities
4. Compare model performance against actual CME transaction prices

> Note: when direct chooser transaction data is unavailable, the project uses a
> historical realized proxy benchmark for reproducible validation.

## 📁 Project Structure

```
chooser-option-pricing/
├── config/                      # Configuration files
│   └── model_params.yaml       # BSM model parameters (Week 3)
├── data/                        # Data storage
│   ├── raw/                     # Raw data from APIs (gitignored)
│   │   ├── yahoo_finance/       # JPM, VIX, dividends
│   │   └── fred/                # Treasury rates (DGS10, etc.)
│   ├── processed/               # Processed dataset (gitignored)
│   └── reports/                 # Data analysis reports (gitignored)
│
├── docs/                        # Documentation
│   ├── feature_engineering.md   # Week 2 feature definitions
│   ├── week4_validation_report.md
│   ├── bsm_benchmark.md
│   ├── week5_ml_architecture.md
│   ├── week6_comparative_analysis.md
│   ├── week7_sensitivity_analysis.md
│   ├── week7_project_timeline.md
│   ├── week8_final_report.md
│   ├── week8_demo_video_script.md
│   ├── week8_presentation_deck.md
│   └── weekly_reports/          # Weekly progress reports
│
├── app/                         # Week 7+ tooling
│   ├── streamlit_app.py         # Week 8 pricing dashboard
│   └── api/
│       └── main.py              # FastAPI service (dual pricing + dashboard)
│
├── .github/workflows/           # CI/CD
│   └── preprocessing.yml       # Data collection + preprocessing pipeline
│
├── models/                      # Trained model files
├── notebooks/                   # Jupyter notebooks
│   ├── week3_bsm_pricing.ipynb  # BSM chooser pricing (Week 3)
│   ├── week3_validation.ipynb   # Validation & sensitivity (Week 3)
│   ├── week4_validation.ipynb   # Baseline BSM error analysis (Week 4)
│   └── week5_ml_frameworks.ipynb # Initial ML pipelines (Week 5)
├── scripts/                     # Automation scripts
│   ├── data_collection/         # Yahoo Finance, FRED collectors
│   ├── analysis/
│   │   └── week7_sensitivity.py # Week 7 SHAP + extreme scenarios
│   └── ml/
│       └── week6_train_eval.py  # Week 6 tuning/evaluation/interpretability
│
├── src/                         # Core pipeline code
│   ├── preprocess.py            # Main preprocessing pipeline (Week 2)
│   ├── data/                    # Data loaders + market_updater (Week 7)
│   ├── features/                # Feature engineering
│   ├── models/                  # BSM chooser pricing module (Week 3)
│   ├── ml/                      # ML datasets, metrics, and model wrappers (Week 5)
│   └── tooling/                 # Week 8 pricing service layer
│
├── tests/                       # Unit tests
├── .env.example                 # API key template
├── requirements.txt
└── README.md
```

### Directory Descriptions

- **src/preprocess.py**, **src/data/**, **src/features/**  
  Data loading, cleaning, feature engineering, and preprocessing pipeline (Week 2).
- **src/models/**  
  BSM chooser option pricing — Monte Carlo simulation + Rubinstein (1991) analytic formula (Week 3).
- **src/ml/**  
  ML utilities for volatility forecasting and end-to-end pricing (datasets, metrics, RF/XGBoost/LR/GBDT/MLP wrappers; Week 5).
- **config/model_params.yaml**  
  Paper parameters (S0, K, r, σ, q, T1, T2) used in BSM replication.
- **notebooks/**  
  Week 3–5 Jupyter notebooks for BSM pricing, validation (Table 3, sensitivity, convergence), baseline evaluation, and initial ML pipelines.
- **scripts/data_collection/**  
  Fetches raw data from Yahoo Finance (no key) and FRED (key required).
- **scripts/analysis/**  
  Week 7 sensitivity (SHAP + stress scenarios).
- **app/**  
  Week 8 tool surface: Streamlit dashboard and FastAPI endpoints.
- **src/data/market_updater.py**  
  Near-real-time Yahoo merge into `data/raw/yahoo_finance/` for UI/API refresh.
- **data/raw/**  
  Raw JPM OHLCV, VIX, dividends, Treasury rates (gitignored).
- **data/processed/**  
  Output of preprocessing: 12+ engineered features, saved as parquet/CSV (gitignored).

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip
- Optional ML dependencies for extended experiments:
  - `xgboost` (tree boosting)
  - `tensorflow` (LSTM)
  - `shap` / `lime` (interpretability, Week 6)

### Installation

```bash
git clone https://github.com/hmit0506/chooser-option-pricing-ml.git
cd chooser-option-pricing-ml
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configuration

- Copy `.env.example` to `.env`
- **FRED_API_KEY**: Required for Treasury data (DGS10). Get a free key at [FRED](https://fred.stlouisfed.org/docs/api/api_key.html). Without it, preprocessing uses a default risk-free rate.
- Yahoo Finance (JPM, VIX) works without any API key.

### Data Pipeline

1. **Collect raw data:**
   ```bash
   python scripts/data_collection/collect_all.py
   ```
   Saves to `data/raw/yahoo_finance/` and `data/raw/fred/`.

2. **Run preprocessing:**
   ```bash
   python src/preprocess.py
   ```
   Produces `data/processed/processed_dataset.parquet` and `.csv` with 12+ engineered features.

3. **Incremental market refresh (Week 7)** — merge latest JPM/VIX into raw Parquet:
   ```python
   from src.data.market_updater import update_market_data_raw
   update_market_data_raw(lookback_days=60)
   ```
   Or use the **Refresh Yahoo raw** button in the Streamlit sidebar / `POST /data/update_market` on the API. After a refresh, restart the app if you need the latest rows inside `load_base_frame()`.

### ML training (required for the dual-pricing tool)

Trained weights are **not** committed to Git (see `.gitignore`: `models/*.pkl`). The Week 8 app (`src/tooling/pricing_tool.py`) loads **`models/week6/best_pricing_model_mlp.pkl`** (or the path recorded in `data/reports/week6/week6_results.json` after a run).

1. **Complete at least the data steps above** so `src/ml/datasets.py::load_base_frame()` can read:
   - `data/raw/yahoo_finance/{TICKER}_daily_ohlcv.parquet` (default `JPM`)
   - `data/raw/yahoo_finance/VIX_daily.parquet`
   - `data/raw/fred/DGS10.parquet` (optional; a fallback rate is used if missing)

2. **Run the Week 6 pipeline** (hyperparameter search, test metrics, SHAP/LIME, saved models — can take significant time):
   ```bash
   python scripts/ml/week6_train_eval.py
   ```
   This writes **`models/week6/*.pkl`**, **`data/reports/week6/`** (JSON, CSV, plots, VIF), etc.

3. **Optional — Week 7 sensitivity artifacts for the Streamlit “Sensitivity” tables:**
   ```bash
   python scripts/analysis/week7_sensitivity.py
   ```
   Requires **`shap`**. Tables read from `data/reports/week7/`; if you skip this step, dual pricing still works after Week 6, but sensitivity sections may be empty.

### Week 8 tool — how to run it correctly

**What the tool does:** `pricing_tool` returns **model-implied** chooser values — Rubinstein/BSM and the trained ML regressor — **not** live CME trade prices. Error bands use historical ML residuals from the loaded context.

**Checklist before `streamlit run`:**

| Step | Purpose |
|------|--------|
| `collect_all.py` | Raw Yahoo (+ FRED if key set) |
| `week6_train_eval.py` | **`.pkl` + `week6_results.json`** (mandatory for ML) |
| `week7_sensitivity.py` | Optional dashboard sensitivity CSVs |
| `.env` / `FRED_API_KEY` | Better rate series for `load_base_frame` |

**Streamlit (browser UI):**

```bash
streamlit run app/streamlit_app.py
```

Open the local URL (usually `http://localhost:8501`). Adjust parameters in the sidebar; review BSM vs ML prices, residual-based bands, and trend charts. Use **Refresh Yahoo raw** / **latest quotes** for near-real-time inputs (still merged into local raw files).

**FastAPI (programmatic access):**

```bash
uvicorn app.api.main:app --reload
```

Interactive docs: **`http://127.0.0.1:8000/docs`**.

**API routes (Week 8):**

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/price/rubinstein` | Closed-form chooser (request body: spot, strike, rates, vol, maturities) |
| `POST` | `/price/dual` | BSM + ML prices + error-margin fields |
| `GET` | `/dashboard/series` | Recent proxy vs BSM vs ML series |
| `GET` | `/dashboard/metrics` | Week 6 benchmark table |
| `GET` | `/dashboard/sensitivity` | Week 7 segmented / calibration tables (if files exist) |
| `POST` | `/data/update_market` | Merge latest Yahoo JPM/VIX into raw storage |
| `GET` | `/data/latest_quotes` | Small equity/VIX snapshot |

### Troubleshooting

Run commands from the **repository root** (the folder that contains `app/`, `src/`, and `config/`).

**1. Streamlit / FastAPI error on startup (`load_tool_context`)**

| Symptom | What to do |
|--------|------------|
| `FileNotFoundError` for `*.pkl` under `models/week6/` | Run `python scripts/ml/week6_train_eval.py` to completion. Confirm the file named in `data/reports/week6/week6_results.json` → `artifacts.best_pricing_model` **exists on disk**. If the repo includes `week6_results.json` from Git but you never trained locally, that path may point to a **missing** pickle — **re-run Week 6** (overwrites JSON + creates `.pkl`) or delete `week6_results.json` and rely on the fallback name `best_pricing_model_mlp.pkl` only after training. |
| `FileNotFoundError` for `JPM_daily_ohlcv.parquet` or `VIX_daily.parquet` | Run `python scripts/data_collection/collect_all.py`. Check `data/raw/yahoo_finance/`. |
| `FileNotFoundError` for `DGS10.parquet` | Optional for `load_base_frame` (uses a default rate). For real rates, set `FRED_API_KEY` in `.env` and re-run collection, or accept the fallback. |
| Empty data / too many NaNs after loading | Yahoo history may be too short for rolling windows (e.g. 252d vol). Use a longer `DATA_START_DATE` in collection if your script supports it, or ensure `collect_all` pulled enough history. |
| Streamlit shows a stale app after data refresh | Restart Streamlit (`Ctrl+C` and `streamlit run ...` again). `@st.cache_resource` caches `load_tool_context()`. |

**2. `week6_train_eval.py` fails**

| Symptom | What to do |
|--------|------------|
| `RuntimeError: shap is required` | `pip install shap` (included in `requirements.txt`; recreate venv if needed). |
| Errors mentioning `xgboost` | `pip install xgboost` (listed in `requirements.txt`). |
| Very long runtime or memory pressure | Hyperparameter search is heavy. Close other apps, run on a machine with more RAM, or temporarily narrow search spaces in the script (advanced). |

**3. Pickle / sklearn version errors when loading `.pkl`**

If `pickle.load` raises errors about missing attributes or incompatible objects, your **scikit-learn** version may differ from the one used to train. Prefer `pip install -r requirements.txt` in a **fresh venv**, then **re-run** `week6_train_eval.py` to regenerate `models/week6/*.pkl`.

**4. `week7_sensitivity.py` fails**

Requires **`shap`**. Needs the same **raw Parquet** inputs as `load_base_frame()` (run **`collect_all.py`** first). The script rebuilds pricing features internally; it does **not** load the Week 6 `.pkl`, but it must be able to read **JPM** and **VIX** history from `data/raw/yahoo_finance/`.

**5. “BSM only” without ML**

The shipped app **does not** support ML-off mode. To use only Rubinstein, call **`POST /price/rubinstein`** on FastAPI, or use `src.models.rubinstein_chooser` in Python, without going through `load_tool_context()`.

### CI/CD

GitHub Actions runs collection + preprocessing on schedule. Add `FRED_API_KEY` as a repository secret (Settings → Secrets and variables → Actions) for full Treasury data. **CI does not train or publish `.pkl` files**; clone + CI alone will not populate `models/week6/`.

## 🔬 Reproducing results

Use this order so dependencies match the internship milestones.

### A. Scripts (exact pipeline / reports)

| Order | Command | Main outputs |
|------|---------|--------------|
| 1 | `python scripts/data_collection/collect_all.py` | `data/raw/**` (gitignored locally) |
| 2 | `python src/preprocess.py` | `data/processed/processed_dataset.*` |
| 3 | `python scripts/ml/week6_train_eval.py` | `models/week6/*.pkl`, `data/reports/week6/**` |
| 4 | `python scripts/analysis/week7_sensitivity.py` | `data/reports/week7/**` |
| 5 | `streamlit run app/streamlit_app.py` / `uvicorn app.api.main:app` | Interactive tool |

Note: **`data/reports/week6/`** and **`data/reports/week7/`** subsets are **tracked** in Git for key CSV/JSON/plots (see `.gitignore` exceptions). Your local run may refresh those files.

### B. Jupyter notebooks (exploration & paper alignment)

Run from the repo root with the same venv and **after** raw data exists (`collect_all.py`). Execution order:

| Notebook | Milestone | What it reproduces |
|----------|-----------|-------------------|
| [notebooks/week3_bsm_pricing.ipynb](notebooks/week3_bsm_pricing.ipynb) | Week 3 | GBM paths, Monte Carlo vs Rubinstein, price ordering (call / chooser / straddle) |
| [notebooks/week3_validation.ipynb](notebooks/week3_validation.ipynb) | Week 3 | Path-by-path Table 3 checks, parameter sensitivities, MC convergence |
| [notebooks/week4_validation.ipynb](notebooks/week4_validation.ipynb) | Week 4 | MAE/RMSE vs realized proxy, regime splits; aligns with `docs/week4_validation_report.md` |
| [notebooks/week5_ml_frameworks.ipynb](notebooks/week5_ml_frameworks.ipynb) | Week 5 | Build datasets, time splits, initial RF/XGB vol + linear/GBDT pricing, metrics vs BSM |

**Authoritative final ML tables and SHAP/LIME exports** come from **`scripts/ml/week6_train_eval.py`**, not from Week 5 notebook defaults. Use the notebook for intuition; use the script for numbers that match `docs/week6_comparative_analysis.md`.

### C. Documentation cross-check

- Baseline metrics and proxy definition: **`docs/week4_validation_report.md`**, **`docs/bsm_benchmark.md`**
- Final ML comparison: **`docs/week6_comparative_analysis.md`**
- Sensitivity narrative: **`docs/week7_sensitivity_analysis.md`**
- Full narrative: **`docs/week8_final_report.md`**

## 📝 Documentation

- [Feature engineering](docs/feature_engineering.md) – 12 features, formulae, rationale
- [Week 2 report](docs/weekly_reports/week2_report.md) – Preprocessing pipeline
- [Week 3 report](docs/weekly_reports/week3_report.md) – BSM model replication & validation
- [Week 4 report](docs/weekly_reports/week4_report.md) – Baseline evaluation weekly summary
- [Week 4 validation report](docs/week4_validation_report.md) – Baseline error metrics and limitations
- [BSM benchmark](docs/bsm_benchmark.md) – Official baseline metrics for ML comparison
- [Week 5 report](docs/weekly_reports/week5_report.md) – ML architecture and pipeline setup
- [Week 5 ML architecture](docs/week5_ml_architecture.md) – Detailed two-approach ML design
- [Week 5 ML frameworks notebook](notebooks/week5_ml_frameworks.ipynb) – Initial train/validation/test pipeline demo
- [Week 6 report](docs/weekly_reports/week6_report.md) – Tuning, final evaluation, and interpretability completion
- [Week 6 comparative analysis](docs/week6_comparative_analysis.md) – Final tuning, ML vs BSM, SHAP/LIME outputs
- [Week 7 report](docs/weekly_reports/week7_report.md) – Sensitivity, tooling, live data
- [Week 7 sensitivity analysis](docs/week7_sensitivity_analysis.md) – SHAP (VIX/sentiment) + extreme scenarios
- [Week 7 timeline & modules](docs/week7_project_timeline.md) – Core modules and commands
- [Week 8 report](docs/weekly_reports/week8_report.md) – Tool completion and final packaging
- [Week 8 final report](docs/week8_final_report.md) – Full project synthesis
- [Week 8 demo script](docs/week8_demo_video_script.md) – 5–10 minute demo runbook
- [Week 8 deck outline](docs/week8_presentation_deck.md) – Final presentation structure

### Repository status

The internship **8-week deliverables** are represented in this repository: data pipeline, BSM/Rubinstein baseline, ML training script, sensitivity analysis, and Week 8 Streamlit/FastAPI tooling. **You must run Week 6 training locally** to use ML-assisted pricing; binary model files are excluded from Git by design.

## 📝 Development Notes

- All code and comments in English (see `.cursorrules`)
- Conventional commits: `feat:`, `fix:`, `docs:`, etc.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.