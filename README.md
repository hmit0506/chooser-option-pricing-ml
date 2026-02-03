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
│   └── (configuration files for API keys, model parameters, etc.)
│
├── data/                    # Data storage
│   ├── raw/                 # Raw data from APIs (gitignored)
│   ├── processed/           # Processed/cleaned data (gitignored)
│   └── reports/             # Data analysis reports (gitignored)
│
├── docs/                    # Documentation
│   └── weekly_reports/      # Weekly progress reports
│
├── logs/                    # Application logs (gitignored)
│
├── models/                  # Trained model files (gitignored)
│   ├── *.pkl, *.h5, *.joblib, etc.
│
├── notebooks/               # Jupyter notebooks for exploration
│
├── scripts/                 # Python scripts
│   ├── analysis/           # Analysis and visualization scripts
│   ├── data_collection/    # Data fetching and preprocessing scripts
│   └── utils/              # Utility functions and helpers
│
├── tests/                   # Unit tests and integration tests
│
├── .cursorignore           # Cursor AI ignore patterns
├── .cursorrules            # Cursor AI coding rules
├── .gitignore              # Git ignore rules
├── LICENSE                 # MIT License
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies
```

### Directory Descriptions

- **config/**: Configuration files for API keys, model hyperparameters, and other settings
- **data/**: All data files (raw, processed, reports) are gitignored to avoid committing large files
- **docs/**: Project documentation and weekly progress reports
- **logs/**: Application logs generated during execution
- **models/**: Trained model files (various formats: pickle, HDF5, joblib, etc.)
- **notebooks/**: Jupyter notebooks for exploratory data analysis and prototyping
- **scripts/**: Production-ready Python scripts organized by functionality
  - **analysis/**: Scripts for data analysis, visualization, and model evaluation
  - **data_collection/**: Scripts for fetching data from APIs (yfinance, Alpha Vantage, FRED)
  - **utils/**: Shared utility functions and helper modules
- **tests/**: Unit tests and integration tests using pytest

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd chooser-option-pricing
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up configuration:
   - Create a `.env` file in the project root (see `.env.example` if available)
   - Add your API keys for data sources (Alpha Vantage, FRED, etc.)

## 📝 Development Notes

- All code and comments must be written in English (see `.cursorrules`)
- Follow PEP 8 style guidelines for Python code
- Write tests for new features in the `tests/` directory
- Use Jupyter notebooks in `notebooks/` for exploratory work
- Commit trained models and data files are excluded via `.gitignore`

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.