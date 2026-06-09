"""
FastAPI serving layer for cross-sell propensity scoring.

Endpoints:
  POST /predict  — score a single customer's vehicle insurance cross-sell propensity
  GET  /health   — liveness check

The model artifact (Pipeline + threshold) is loaded once at startup via the
FastAPI lifespan hook and shared across all requests through _store.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.model import load_model
from src.predict import predict_single


# ── Request / Response schemas ────────────────────────────────────────────────

class CustomerRecord(BaseModel):
    """
    Raw customer fields for a single cross-sell scoring request.

    All values must be pre-encoded (numeric) — matching the output of
    src.data.clean(). String fields (Gender, Vehicle_Age, Vehicle_Damage)
    should be encoded as described below before sending.
    """

    Gender:                int   = Field(..., ge=0, le=1,
                                         description="0=Female, 1=Male")
    Age:                   int   = Field(..., ge=18, le=100)
    Driving_License:       int   = Field(..., ge=0, le=1)
    Region_Code:           int   = Field(..., ge=0)
    Previously_Insured:    int   = Field(..., ge=0, le=1)
    Vehicle_Age:           int   = Field(..., ge=0, le=2,
                                         description="0=<1yr, 1=1-2yr, 2=>2yr")
    Vehicle_Damage:        int   = Field(..., ge=0, le=1,
                                         description="0=No, 1=Yes")
    Annual_Premium:        float = Field(..., ge=0)
    Policy_Sales_Channel:  int   = Field(..., ge=0)
    Vintage:               int   = Field(..., ge=0)


class PredictRequest(BaseModel):
    """Scoring request body — one customer record."""
    customer: CustomerRecord


class PredictResponse(BaseModel):
    """Scoring response — probability, binary prediction, and threshold used."""
    probability: float = Field(..., description="Predicted conversion probability [0, 1]")
    prediction:  int   = Field(..., description="1 = likely to convert, 0 = unlikely")
    threshold:   float = Field(..., description="Decision threshold used")


class HealthResponse(BaseModel):
    """Liveness check response."""
    model_config = {"protected_namespaces": ()}

    status: str
    model_loaded: bool


# ── App setup ─────────────────────────────────────────────────────────────────

_store: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifact on startup; clear on shutdown."""
    _store["artifact"] = load_model("models/cross_sell_model.pkl")
    print(f"Model loaded — threshold={_store['artifact']['threshold']:.4f}")
    yield
    _store.clear()


app = FastAPI(
    title="Cross-Sell Propensity API",
    description=(
        "Scores a customer's likelihood to purchase vehicle insurance "
        "cross-sell. Returns probability, binary prediction, and decision threshold."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """
    Liveness check.

    Returns 200 with model_loaded=True once the model artifact has been
    loaded successfully at startup. Use this to confirm the service is
    ready before sending scoring requests.
    """
    return HealthResponse(
        status="ok",
        model_loaded="artifact" in _store,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """
    Score a single customer's cross-sell propensity.

    Accepts a cleaned customer record (numeric fields only — see
    CustomerRecord for encoding details) and returns a conversion
    probability, binary prediction, and the decision threshold used.

    Raises:
        503 if the model artifact has not been loaded yet.
        422 if the input fails feature engineering or pipeline inference.
    """
    if "artifact" not in _store:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        result = predict_single(
            customer=request.customer.model_dump(),
            artifact=_store["artifact"],
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return PredictResponse(**result)
