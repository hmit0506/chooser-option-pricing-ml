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

## 📁 Project Structure

```
chooser-option-pricing/
├── config/                  # Configuration files
│   └── model_params.yaml   # BSM model parameters (Week 3)
├── data/                    # Data storage
│   ├── raw/                 # Raw data from APIs (gitignored)
│   │   ├── yahoo_finance/   # JPM, VIX, dividends
│   │   └── fred/            # Treasury rates (DGS10, etc.)
│   ├── processed/           # Processed dataset (gitignored)
│   └── reports/             # Data analysis reports (gitignored)
│
├── docs/                    # Documentation
│   ├── feature_engineering.md
│   └── weekly_reports/      # Weekly progress reports
│
├── .github/workflows/       # CI/CD
│   └── preprocessing.yml   # Data collection + preprocessing pipeline
│
├── models/                  # Trained model files (gitignored)
├── notebooks/               # Jupyter notebooks
│   ├── week3_bsm_pricing.ipynb   # BSM chooser pricing (Week 3)
│   └── week3_validation.ipynb    # Validation & sensitivity (Week 3)
├── scripts/                 # Data collection scripts
│   ├── data_collection/    # Yahoo Finance, FRED collectors
│   ├── analysis/
│   └── utils/
│
├── src/                     # Core pipeline code
│   ├── preprocess.py       # Main preprocessing pipeline (Week 2)
│   ├── data/               # Data loaders
│   ├── features/           # Feature engineering
│   └── models/             # BSM chooser pricing module (Week 3)
│
├── tests/                   # Unit tests
├── .env.example             # API key template
├── requirements.txt
└── README.md
```

### Directory Descriptions

- **src/models/**: BSM chooser option pricing — Monte Carlo simulation + Rubinstein (1991) analytic formula
- **src/data/**, **src/features/**, **src/preprocess.py**: Data loading, feature engineering, preprocessing pipeline
- **config/model_params.yaml**: Paper parameters (S0, K, r, σ, q, T1, T2)
- **notebooks/**: Week 3 BSM pricing and validation notebooks with sensitivity analysis
- **scripts/data_collection/**: Fetches raw data from Yahoo Finance (no key) and FRED (key required)
- **data/raw/**: Raw JPM OHLCV, VIX, dividends, Treasury rates
- **data/processed/**: Output of preprocessing: 12+ features, parquet + CSV

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

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

### CI/CD

GitHub Actions runs collection + preprocessing on schedule. Add `FRED_API_KEY` as a repository secret (Settings → Secrets and variables → Actions) for full Treasury data.

## 📝 Documentation

- [Feature engineering](docs/feature_engineering.md) – 12 features, formulae, rationale
- [Week 2 report](docs/weekly_reports/week2_report.md) – Preprocessing pipeline
- [Week 3 report](docs/weekly_reports/week3_report.md) – BSM model replication & validation

## 📝 Development Notes

- All code and comments in English (see `.cursorrules`)
- Conventional commits: `feat:`, `fix:`, `docs:`, etc.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.