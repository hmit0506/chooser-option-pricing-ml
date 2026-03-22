"""
FastAPI skeleton for chooser pricing and health checks (Week 7).

Run locally:
  uvicorn app.api.main:app --reload --app-dir .
"""

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


app = FastAPI(title="Chooser Option Pricing API", version="0.1.0")


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


@app.post("/data/update_market")
def update_market(lookback_days: int = 60) -> dict:
    """Trigger raw Yahoo merge (JPM + VIX)."""
    return update_market_data_raw(lookback_days=lookback_days)


@app.get("/data/latest_quotes")
def latest_quotes() -> dict:
    return get_latest_quote_summary()
