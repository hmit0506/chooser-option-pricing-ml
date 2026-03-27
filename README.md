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

### Tool apps (Week 8)

- **Streamlit UI (dual pricing + error bands + dashboard):**
  ```bash
  streamlit run app/streamlit_app.py
  ```
- **FastAPI:**
  ```bash
  uvicorn app.api.main:app --reload
  ```
- **Week 7/8 sensitivity report generation:**
  ```bash
  python scripts/analysis/week7_sensitivity.py
  ```

### API snapshot (Week 8)

- `POST /price/rubinstein`
- `POST /price/dual`
- `GET /dashboard/series`
- `GET /dashboard/metrics`
- `GET /dashboard/sensitivity`
- `POST /data/update_market`
- `GET /data/latest_quotes`

### CI/CD

GitHub Actions runs collection + preprocessing on schedule. Add `FRED_API_KEY` as a repository secret (Settings → Secrets and variables → Actions) for full Treasury data.

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

### Current Phase

- Week 8: Tool feature completion + final report/demo/deck preparation
- Next: final polish, recording, and submission

## 📝 Development Notes

- All code and comments in English (see `.cursorrules`)
- Conventional commits: `feat:`, `fix:`, `docs:`, etc.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.