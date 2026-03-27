"""
Streamlit pricing tool (Week 8):
- Dual pricing: Rubinstein BSM + best ML model
- Error margin display
- Dashboard: trends, sensitivity slices, model metrics
"""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.market_updater import get_latest_quote_summary, update_market_data_raw
from src.tooling.pricing_tool import (
    dashboard_series,
    dual_price,
    load_tool_context,
    performance_metrics,
    sensitivity_tables,
)


def main() -> None:
    st.set_page_config(page_title="Chooser Option Pricing", layout="wide")
    st.title("Chooser Option Pricing Tool")
    st.caption("Week 8: dual pricing, error margins, and interactive dashboard.")

    @st.cache_resource
    def _ctx():
        return load_tool_context()

    ctx = _ctx()
    defaults = ctx.config
    latest = ctx.frame.iloc[-1]

    with st.sidebar:
        st.header("Parameters")
        s0 = st.number_input("Spot S", value=float(latest["close"]), format="%.4f")
        k = st.number_input("Strike K", value=float(defaults["k"]), format="%.4f")
        r = st.number_input("Risk-free r (annual)", value=float(latest["r"]), format="%.6f")
        q = st.number_input("Dividend yield q (annual)", value=float(defaults["q"]), format="%.6f")
        sigma = st.number_input("Volatility σ (annual)", value=float(latest["sigma_252d"]), format="%.6f")
        vix = st.number_input("VIX level", value=float(latest["vix"]), format="%.4f")
        sentiment = st.slider("Sentiment proxy (0-1)", min_value=0.0, max_value=1.0, value=float(latest["sentiment_proxy"]), step=0.01)
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
        st.caption("Tip: refresh and then rerun app for newest dashboard trend data.")

    price = dual_price(
        ctx=ctx,
        s=s0,
        k=k,
        r=r,
        q=q,
        sigma=sigma,
        t1=t1,
        t2=t2,
        vix=vix,
        sentiment=sentiment,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("BSM/Rubinstein price", f"{price['bsm_price']:.4f}")
    with col2:
        st.metric("Best ML price", f"{price['ml_price']:.4f}")
    with col3:
        st.metric("ML - BSM", f"{price['delta_ml_minus_bsm']:.4f}")

    st.subheader("Error Margin (based on historical ML residuals)")
    c1, c2, c3 = st.columns(3)
    c1.metric("68% band", f"+/- {price['error_margin_68']:.4f}")
    c2.metric("95% band", f"+/- {price['error_margin_95']:.4f}")
    c3.metric("Historical MAE", f"{price['historical_residual_mae']:.4f}")

    st.divider()
    st.subheader("Price Trend Dashboard")
    trend = dashboard_series(ctx, n_points=200).reset_index().rename(columns={"index": "date"})
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trend["date"], y=trend["target"], name="Proxy actual", mode="lines"))
    fig.add_trace(go.Scatter(x=trend["date"], y=trend["bsm_price"], name="BSM", mode="lines"))
    fig.add_trace(go.Scatter(x=trend["date"], y=trend["ml_price"], name="ML", mode="lines"))
    fig.update_layout(height=420, legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    err_fig = px.line(
        trend,
        x="date",
        y=["ml_abs_err", "bsm_abs_err"],
        labels={"value": "Absolute error", "variable": "Model"},
        title="Rolling absolute error comparison",
    )
    st.plotly_chart(err_fig, use_container_width=True)

    st.subheader("Performance Metrics (Week 6 benchmark)")
    metrics = performance_metrics()
    st.dataframe(metrics, use_container_width=True, hide_index=True)

    st.subheader("Sensitivity Dashboard (Week 7 follow-up)")
    sens = sensitivity_tables()
    if "by_maturity" in sens:
        st.markdown("**VIX/sentiment SHAP by maturity bucket**")
        st.dataframe(sens["by_maturity"], use_container_width=True, hide_index=True)
    if "by_moneyness" in sens:
        st.markdown("**VIX/sentiment SHAP by moneyness bucket**")
        st.dataframe(sens["by_moneyness"], use_container_width=True, hide_index=True)
    if "historical_event_calibration" in sens:
        st.markdown("**Historical extreme-event calibration (2020/2022)**")
        st.dataframe(sens["historical_event_calibration"], use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
