# Feature Engineering Documentation

**Project:** Avalok Capital Chooser Option Pricing Model  
**Week:** 2  
**Version:** 1.0  

## Overview

This document describes the features engineered for the preprocessing pipeline. All features are designed for use in BSM chooser option pricing and ML-based volatility prediction, with **no look-ahead bias** (rolling windows use only historical data).

---

## Feature Table

| # | Feature Name | Type | Calculation Method | Rationale |
|---|--------------|------|---------------------|-----------|
| 1 | `log_return` | Traditional | \(\ln(\text{Close}_t / \text{Close}_{t-1})\) | Standard log returns for BSM; time-additive, symmetric. |
| 2 | `vol_21d` | Traditional | Rolling 21-day std of `log_return` × √252 | Short-term (≈1 month) annualized realized volatility. |
| 3 | `vol_63d` | Traditional | Rolling 63-day std of `log_return` × √252 | Medium-term (~3 months) annualized volatility. |
| 4 | `vol_252d` | Traditional | Rolling 252-day std of `log_return` × √252 | Annual realized volatility; core BSM vol input. |
| 5 | `vix_close` | Traditional | VIX index daily close (forward-filled) | Market implied volatility proxy; addresses constant-vol limitation. |
| 6 | `treasury_10y` | Traditional | 10Y Treasury yield (DGS10) as decimal | Risk-free rate for BSM; converted from % to decimal. |
| 7 | `dividend_yield_proxy` | Traditional | Rolling 252d sum of dividends / Close | Dividend yield for BSM; handles sparse dividend data. |
| 8 | `volume_ma_21d` | Traditional | 21-day rolling mean of Volume | Liquidity and trading activity; regime indicator. |
| 9 | `high_low_range_pct` | Traditional | (High − Low) / Close × 100 | Intraday volatility proxy; complements close-to-close vol. |
| 10 | `vix_jpm_corr_63d` | Advanced | Rolling 63d correlation(JPM returns, VIX pct change) | Volatility regime; hedge ratio; negative when stock and vol move opposite. |
| 11 | `treasury_momentum_21d` | Advanced | 21-day change in Treasury 10Y yield | Interest rate trend; affects discounting and volatility. |
| 12 | `sentiment_proxy` | Advanced | 1 − minmax_norm(VIX) over 252d rolling window | Placeholder for sentiment (0–1 scale); high VIX → low sentiment. Full NewsAPI/VADER later. |

---

## Data Sources

| Source Column | Origin | Preprocessing |
|---------------|--------|---------------|
| Close, High, Low, Open, Volume, Dividends | Yahoo Finance (JPM) | Time-aligned, forward-filled on gaps |
| VIX Close | Yahoo Finance (^VIX) | Forward-filled |
| DGS10 | FRED (10Y Treasury) | Converted %→decimal, forward-filled |
| Dividend amounts | Yahoo Finance (JPM dividends) | Expanded to daily (0 on non-dividend days) |

---

## Implementation Notes

1. **Rolling windows**: All rolling calculations use `min_periods` to handle warm-up; initial rows with insufficient history are dropped before saving.
2. **Annualization**: Volatilities use √252 (trading days per year).
3. **Outlier handling**: IQR winsorization (factor=1.5) applied to feature columns post-engineering.
4. **Missing values**: Linear interpolation (limit=5) followed by forward/backward fill.
5. **Time alignment**: Base index = union of JPM and VIX trading days; FRED/VIX forward-filled to match.

---

## Column Count

The processed dataset includes **≥10 feature columns** as required: 12 features total, plus base columns (Open, High, Low, Close, Volume, Dividends) for reproducibility.
