"""
Streamlit prototype for chooser option pricing (Week 7).

Run from project root:
  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.market_updater import get_latest_quote_summary, update_market_data_raw
from src.models import rubinstein_chooser

CONFIG_PATH = PROJECT_ROOT / "config" / "model_params.yaml"


def load_defaults() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["model"]


def main() -> None:
    st.set_page_config(page_title="Chooser Option Pricing", layout="wide")
    st.title("Chooser Option Pricing — Prototype")
    st.caption("Rubinstein (1991) closed-form chooser vs BSM parameters. Week 7 UI shell.")

    defaults = load_defaults()

    with st.sidebar:
        st.header("Parameters")
        s0 = st.number_input("Spot S", value=float(defaults["s0"]), format="%.4f")
        k = st.number_input("Strike K", value=float(defaults["k"]), format="%.4f")
        r = st.number_input("Risk-free r (annual)", value=float(defaults["r"]), format="%.6f")
        q = st.number_input("Dividend yield q (annual)", value=float(defaults["q"]), format="%.6f")
        sigma = st.number_input("Volatility σ (annual)", value=float(defaults["sigma"]), format="%.6f")
        t1 = st.number_input("Decision T1 (years)", value=float(defaults["t1"]), format="%.4f")
        t2 = st.number_input("Maturity T2 (years)", value=float(defaults["t2"]), format="%.4f")
        st.divider()
        st.subheader("Market data")
        if st.button("Refresh Yahoo raw (JPM + VIX)"):
            with st.spinner("Downloading and merging..."):
                try:
                    res = update_market_data_raw(lookback_days=60)
                    st.success(f"Updated: equity rows={res['equity_rows_total']}, VIX rows={res['vix_rows_total']}")
                    st.json(res)
                except Exception as e:  # pragma: no cover
                    st.error(str(e))
        if st.button("Show latest quote snapshot"):
            try:
                snap = get_latest_quote_summary()
                st.json(snap)
            except Exception as e:  # pragma: no cover
                st.error(str(e))

    col1, col2 = st.columns(2)
    with col1:
        price = rubinstein_chooser(s0, k, r, q, sigma, t1, t2)
        st.metric("Rubinstein chooser price", f"{price:.4f}")

        vol_shock = rubinstein_chooser(s0, k, r, q, sigma * 1.5, t1, t2)
        rate_shock = rubinstein_chooser(s0, k, r + 0.02, q, sigma, t1, t2)
        both = rubinstein_chooser(s0, k, r + 0.02, q, sigma * 1.5, t1, t2)
        st.subheader("Quick stress (same as Week 7 report)")
        st.write(
            {
                "baseline": price,
                "vol_x1.5 (+50%)": vol_shock,
                "r + 200 bps": rate_shock,
                "both": both,
            }
        )

    with col2:
        st.subheader("Notes")
        st.markdown(
            """
            - **Production API**: see `app/api/main.py` (FastAPI).
            - **Sensitivity report**: `docs/week7_sensitivity_analysis.md`.
            - **Data refresh**: `src/data/market_updater.py`.
            """
        )


if __name__ == "__main__":
    main()
