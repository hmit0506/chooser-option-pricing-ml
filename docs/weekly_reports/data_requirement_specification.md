# Data Requirement Specification

**Project:** Avalok Capital Quantitative Research & Trading - Advanced Chooser Option Pricing Model with Real-World Data & Machine Learning  
**Author:** Shi Qian  
**Date:** February 3, 2026  
**Week:** 1  
**Version:** 1.0  

## Introduction
This document defines the scope of data required for the 8-week internship project at Avalok Capital. The project extends the BSM-based chooser option pricing model by integrating real-time/historical data and ML to improve accuracy under dynamic market conditions (e.g., stochastic volatility as discussed in Hull & White, 1987). 

Data will be collected from financial, macroeconomic, and sentiment sources to enrich features beyond basic BSM parameters (stock price, strike, volatility, risk-free rate, dividends). This supports objectives like volatility prediction (Objective 3) and performance comparison (Objective 4). The time period is January 1, 2018, to February 3, 2026, with daily frequency where possible. Initial focus is on JPMorgan (JPM) stock as per the outline.

The data will enable ≥10 features by Week 2 (e.g., rolling volatilities, sentiment scores) for ML models (LSTM/XGBoost for volatility, end-to-end NN for pricing).

## Data Categories

### 1. Financial Data
Financial data provides core inputs for BSM pricing and feature engineering (e.g., returns, volatilities). Focus on JPM-specific and market volatility metrics.

| Data Point | Description | Source | Format | Period | Why Needed |
|------------|-------------|--------|--------|--------|------------|
| JPM Daily OHLCV | Open, High, Low, Close, Volume, Adjusted Close for JPM stock. | Yahoo Finance / Alpha Vantage API | Time-series CSV/Parquet | 2018-01-01 to 2026-02-03 (daily) | Calculate daily returns, rolling volatilities; BSM stock price input. |
| JPM Dividends | Dividend payments and growth rates. | Yahoo Finance | CSV | Same as above | BSM dividend yield input; feature for dividend growth. |
| VIX Index | CBOE Volatility Index levels (daily close). | Yahoo Finance (ticker: ^VIX) | CSV | Same as above | Proxy for market volatility; feature for VIX-JPM correlation. |
| CME Option Prices | Actual JPM chooser/related option transaction prices (for validation). | CME Group Historical Data (manual/API if available) | CSV | Subset for benchmark periods | Baseline for error metrics (MAE/RMSE) in Week 4. |

**Scope Notes:** At least 5 features from this category (e.g., daily returns, 30-day vol). Handle non-trading days with forward-fill.

### 2. Macroeconomic Data
Macro data captures broader economic factors affecting option pricing (e.g., interest rates for BSM risk-free rate).

| Data Point | Description | Source | Format | Period | Why Needed |
|------------|-------------|--------|--------|--------|------------|
| Treasury Yields | 10-Year US Treasury Constant Maturity Rate (proxy for risk-free rate). | FRED API (series: DGS10) | Time-series CSV | 2018-01-01 to 2026-02-03 (daily) | BSM risk-free rate input; feature for interest rate momentum. |
| Inflation Indicators | CPI or PCE inflation rates (monthly, interpolate to daily). | FRED API (e.g., CPIAUCSL) | CSV | Same as above | Contextual for regime-dependent risks (e.g., post-FOMC). |
| GDP Growth | Quarterly GDP growth rates (interpolated). | FRED API (e.g., GDP) | CSV | Same as above | Macro feature for ML volatility prediction. |

**Scope Notes:** 3-4 features (e.g., rate momentum). Align timestamps with financial data.

### 3. Sentiment Data
Sentiment data addresses BSM's constant volatility flaw by incorporating market mood (e.g., news impact on volatility spikes).

| Data Point | Description | Source | Format | Period | Why Needed |
|------------|-------------|--------|--------|--------|------------|
| News Headlines/Sentiment | Raw headlines/articles about JPM/finance; scored 0-1 (positive/negative). | NewsAPI (keywords: "JPMorgan" OR "JPM") | JSON/CSV (text + scores via NLP) | 2018-01-01 to 2026-02-03 (daily aggregated) | Feature for sentiment scores; ML input for volatility (e.g., post-earnings). |
| Social Media Mentions | X/Twitter mentions of JPM (volume/sentiment). | Optional: X API if accessible (else proxy via NewsAPI) | CSV | Same as above | Advanced feature for sentiment impact gaps (Week 4 analysis). |

**Scope Notes:** Process raw text with libraries like NLTK/VADER for scores (in Week 2). Aim for 2-3 features.

## Collection Plan
- **Tools/APIs:** Yahoo Finance (yfinance lib), Alpha Vantage (free key), FRED (free key), NewsAPI (free alternative to Reuters).
- **Frequency:** Daily pulls for real-time; historical batch in Week 1.
- **Automation:** Scripts in Python (Pandas for handling); schedule via GitHub Actions (Week 2).
- **Storage:** Raw in data/raw/ (CSV/Parquet); git-ignore large files.

## Risks and Mitigation
- **API Limits:** Alpha Vantage: 25 calls/day – use Yahoo as primary; batch requests.
- **Missing Data:** Interpolate (linear) in Week 2; log gaps.
- **Data Quality:** Validate against multiple sources; handle outliers (IQR).
- **Privacy/Compliance:** Use public data only; no proprietary Avalok info.

## References
- Project Outline: Avalok Capital PDF.
- Hull, J. C., & White, A. (1987). Pricing of Options on Assets with Stochastic Volatilities.
- Gu, S., et al. (2020). Empirical Asset Pricing with Machine Learning.