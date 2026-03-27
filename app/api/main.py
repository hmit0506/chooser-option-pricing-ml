"""FastAPI service for pricing + dashboard data (Week 8)."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml
from fastapi import FastAPI
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.market_updater import get_latest_quote_summary, update_market_data_raw
from src.models import rubinstein_chooser
from src.tooling.pricing_tool import (
    dashboard_series,
    dual_price,
    load_tool_context,
    performance_metrics,
    sensitivity_tables,
)

CONFIG_PATH = PROJECT_ROOT / "config" / "model_params.yaml"


def _load_model_defaults() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["model"]


class RubinsteinRequest(BaseModel):
    s: float = Field(..., description="Spot price")
    k: float = Field(..., description="Strike")
    r: float = Field(..., description="Risk-free rate (annual)")
    q: float = Field(..., description="Dividend yield (annual)")
    sigma: float = Field(..., description="Volatility (annual)")
    t1: float = Field(..., description="Chooser decision time (years)")
    t2: float = Field(..., description="Option maturity (years)")


class RubinsteinResponse(BaseModel):
    price: float
    model: str = "rubinstein_chooser"


class DualPriceRequest(BaseModel):
    s: float
    k: float
    r: float
    q: float
    sigma: float
    t1: float
    t2: float
    vix: float
    sentiment: float = Field(..., ge=0.0, le=1.0)


app = FastAPI(title="Chooser Option Pricing API", version="0.1.0")
_CTX = load_tool_context()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "chooser-pricing"}


@app.get("/config/defaults")
def config_defaults() -> dict:
    return _load_model_defaults()


@app.post("/price/rubinstein", response_model=RubinsteinResponse)
def price_rubinstein(body: RubinsteinRequest) -> RubinsteinResponse:
    p = rubinstein_chooser(body.s, body.k, body.r, body.q, body.sigma, body.t1, body.t2)
    return RubinsteinResponse(price=float(p))


@app.post("/price/dual")
def price_dual(body: DualPriceRequest) -> dict:
    return dual_price(
        ctx=_CTX,
        s=body.s,
        k=body.k,
        r=body.r,
        q=body.q,
        sigma=body.sigma,
        t1=body.t1,
        t2=body.t2,
        vix=body.vix,
        sentiment=body.sentiment,
    )


@app.post("/data/update_market")
def update_market(lookback_days: int = 60) -> dict:
    """Trigger raw Yahoo merge (JPM + VIX)."""
    return update_market_data_raw(lookback_days=lookback_days)


@app.get("/data/latest_quotes")
def latest_quotes() -> dict:
    return get_latest_quote_summary()


@app.get("/dashboard/series")
def get_dashboard_series(n_points: int = 200) -> list[dict]:
    df = dashboard_series(_CTX, n_points=n_points).reset_index().rename(columns={"index": "date"})
    df["date"] = df["date"].astype(str)
    return df.to_dict(orient="records")


@app.get("/dashboard/metrics")
def get_metrics() -> list[dict]:
    return performance_metrics().to_dict(orient="records")


@app.get("/dashboard/sensitivity")
def get_sensitivity() -> dict:
    tabs = sensitivity_tables()
    return {k: v.to_dict(orient="records") for k, v in tabs.items()}
